import asyncio
import cv2
import numpy as np
import json
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame
from server.shm_manager import ShmManager


class VideoProcessorTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, mode="grid"):
        super().__init__()
        self.mode = mode
        # 挂载共享内存
        self.shm_manager = ShmManager()
        self.frame_buffer = self.shm_manager.attach()

    async def recv(self):
        timestamp, pts = await self.next_timestamp()

        # 1. 零拷贝读取 Flask 写入的原始帧
        img = self.frame_buffer.copy()

        # 2. 核心：九宫格处理 (3x3 布局)
        if self.mode == "grid":
            # 缩小到 213x160 以拼成 640x480 的九宫格
            small = cv2.resize(img, (213, 160))

            # 模拟 9 个不同的识别窗口（此处应替换为你真实的 9 种渲染逻辑）
            cells = []
            for i in range(9):
                temp = small.copy()
                cv2.putText(temp, f"Type {i + 1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cells.append(temp)

            # 拼接：3行3列
            row1 = np.hstack(cells[0:3])
            row2 = np.hstack(cells[3:6])
            row3 = np.hstack(cells[6:9])
            # 修正宽度微调导致的尺寸不匹配
            final_img = cv2.resize(np.vstack([row1, row2, row3]), (640, 480))
        else:
            final_img = img

        # 3. 封装为 WebRTC 帧
        frame = VideoFrame.from_ndarray(final_img, format="bgr24")
        frame.pts = pts
        frame.time_base = timestamp
        print("DEBUG: Generating Frame")
        return frame


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()

    track = VideoProcessorTrack(mode=params.get("mode", "grid"))
    pc.addTrack(track)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    )


if __name__ == "__main__":
    app = web.Application()
    app.router.add_post("/offer", offer)
    print("🚀 处理端 WebRTC 服务已启动在 8888 端口")
    web.run_app(app, port=8888)