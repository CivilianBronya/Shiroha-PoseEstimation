import asyncio
import cv2
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame
from multiprocessing import shared_memory

# 共享内存配置（与 Flask 端一致）
SHM_NAME = "shiroha_frame"
SHAPE = (480, 640, 3)


class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, mode="grid"):
        super().__init__()
        self.mode = mode
        # 挂载共享内存
        try:
            self.existing_shm = shared_memory.SharedMemory(name=SHM_NAME)
            self.frame_buffer = np.ndarray(SHAPE, dtype=np.uint8, buffer=self.existing_shm.buf)
        except Exception as e:
            print(f"❌ 共享内存挂载失败，请确保 Flask 项目已启动并创建内存: {e}")
            self.frame_buffer = np.zeros(SHAPE, dtype=np.uint8)

    async def recv(self):
        """
        WebRTC 的核心：每一帧渲染都在这里触发
        """
        timestamp, pts = await self.next_timestamp()

        # 1. 零拷贝读取 Flask 写入的原始帧
        raw_img = self.frame_buffer.copy()

        # 2. 核心：根据模式进行九种处理
        # 这里的处理应该尽量轻量化，如果模式很多，建议在此处引入并行逻辑
        processed_img = self.process_by_mode(raw_img, self.mode)

        # 3. 封装为 av.VideoFrame (aiortc 需要的格式)
        # H.264 编码器会自动比对前一帧，只发送差异部分（类似 VNC 原理）
        frame = VideoFrame.from_ndarray(processed_img, format="bgr24")
        frame.pts = pts
        frame.time_base = timestamp
        return frame

    def process_by_mode(self, frame, mode):
        # 此处集成你 main.py 中的 9 种识别逻辑
        # 如果 mode == 'grid'，执行九宫格拼接
        if mode == "grid":
            # 简化示例：将原图缩小并复制成 2x2（你可以改为 3x3）
            small = cv2.resize(frame, (320, 240))
            h1 = np.hstack([small, small])
            h2 = np.hstack([small, small])
            return np.vstack([h1, h2])
        return frame


# --- WebRTC 信令处理 (供 Flask 调用) ---
async def create_answer(offer_sdp, offer_type, mode):
    pc = RTCPeerConnection()

    # 创建轨道
    track = VideoTransformTrack(mode=mode)
    pc.addTrack(track)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type=offer_type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return pc.localDescription.sdp, pc.localDescription.type