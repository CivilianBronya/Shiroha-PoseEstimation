# -*- coding: utf-8 -*-
import asyncio
import cv2
import numpy as np
import json
import logging
import fractions
import aiohttp
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame
from concurrent.futures import ThreadPoolExecutor

# --- 导入业务组件 ---
from server.shm_manager import ShmManager
from pose.body_pose import BodyPose
from face.head_pose import HeadPose
from rig.skeleton import SkeletonSolver
from rig.face_solver import FaceSolver, MODE_SINGLE, MODE_MULTI

from render.single_stickman_renderer import SingleStickmanRenderer
from render.multi_stickman_renderer import MultiStickmanRenderer
from render.fall_detector_renderer import FallDetectorRenderer
from render.face_recognition_renderer import FaceRecognitionRenderer
from render.intrusion_detection_renderer import IntrusionDetectionRenderer
from render.loitering_detection_renderer import LoiteringDetectionRenderer
from render.static_detection_renderer import StaticDetectionRenderer
from render.vigorous_activity_renderer import VigorousActivityRenderer
from render.activity_level_renderer import ActivityLevelRenderer
from analysis.fall_detector import FallDetector

# --- 基础配置 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rtc_main")
# 9路并行，线程池设为 12-15 保证 CPU 调度顺滑
executor = ThreadPoolExecutor(max_workers=15)

# 读取配置文件
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# --- 1. 全局初始化 AI 组件 (单例模式) ---
logger.info("🚀 正在预加载 AI 推理内核...")

body_single = BodyPose(num_poses=config["pose"]["single"]["num_poses"])
body_multi = BodyPose(num_poses=config["pose"]["multi"]["num_poses"])
head_model = HeadPose()
solver = SkeletonSolver(filter_alpha=config["skeleton"]["filter_alpha"])
single_stick_renderer = SingleStickmanRenderer()

# 渲染器实例化
fall_detector_logic = FallDetector(ground_threshold_sec=config["fall_detector"]["ground_threshold_sec"])
fall_renderer = FallDetectorRenderer(fall_detector_logic)
pose_multi_renderer = MultiStickmanRenderer()

intrusion_renderer = IntrusionDetectionRenderer()
loitering_renderer = LoiteringDetectionRenderer(
    alert_duration=config["detection"]["loitering"]["alert_duration"],
    cycle_length=config["detection"]["loitering"]["cycle_length"],
    alert_threshold=config["detection"]["loitering"]["alert_threshold"]
)
static_renderer = StaticDetectionRenderer(
    history_length=config["detection"]["static"]["history_length"],
    movement_threshold=config["detection"]["static"]["movement_threshold"]
)
vigorous_renderer = VigorousActivityRenderer(
    activity_threshold=config["detection"]["vigorous_activity"]["activity_threshold"]
)
activity_renderer = ActivityLevelRenderer(
    low_threshold=config["detection"]["activity_level"]["low_threshold"],
    high_threshold=config["detection"]["activity_level"]["high_threshold"]
)

face_solver_multi = FaceSolver(
    filter_alpha=config["face"]["filter_alpha"],
    min_area=config["face"]["min_area"],
    mode=MODE_MULTI
)
face_render_node = FaceRecognitionRenderer(face_solver_multi)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# --- 2. 视频流处理轨道 ---
class VideoProcessorTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, mode="type1", shm_name="shiroha_frame"):
        super().__init__()
        self.mode = mode.lower()  # 确保匹配 type1, type2...
        self.shm_name = shm_name
        self.shm_manager = ShmManager(name=shm_name)
        self.frame_buffer = None
        self._timestamp = 0
        self._fps = 15  # 矩阵模式下 15 帧最稳
        self._clock_rate = 90000

        try:
            self.frame_buffer = self.shm_manager.attach()
            logger.info(f"✅ [Track-{mode}] Attached to {shm_name}")
        except Exception as e:
            logger.error(f"❌ [Track-{mode}] SHM Error: {e}")

    async def recv(self):
        await asyncio.sleep(1 / self._fps)
        pts = self._timestamp
        self._timestamp += int(self._clock_rate / self._fps)

        loop = asyncio.get_event_loop()
        try:
            # 这里的推理必须在线程池运行，否则会阻塞 WebRTC 事件循环
            final_img = await loop.run_in_executor(executor, self._process_frame)
        except Exception as e:
            logger.error(f"Render Error: {e}")
            final_img = np.zeros((480, 640, 3), dtype=np.uint8)

        frame = VideoFrame.from_ndarray(final_img, format="bgr24")
        frame.pts = pts
        frame.time_base = fractions.Fraction(1, self._clock_rate)
        return frame

    def _process_frame(self):
        if self.frame_buffer is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. 拷贝原始帧
        raw_frame = self.frame_buffer.copy()
        draw_frame = raw_frame.copy()

        try:
            # --- Type 1: Fall Detector (单人) ---
            if self.mode == "type1":
                res = body_single.detect(raw_frame)
                if res and 'landmark_points' in res:
                    pts = res['landmark_points']
                    skel = solver.solve({i: pt for i, pt in enumerate(pts)}, head_model.detect(raw_frame))
                    draw_frame = single_stick_renderer.draw(draw_frame, pts, skel)
                    draw_frame = fall_renderer.draw(draw_frame, skel)

            # --- Type 2: Pose Monitoring (多人) ---
            elif self.mode == "type2":
                res = body_multi.detect(raw_frame)
                if res and 'people' in res:
                    multi_data = [p['landmark_points'] for p in res['people'] if 'landmark_points' in p]
                    draw_frame = pose_multi_renderer.draw(draw_frame, multi_data)

            # --- Type 3: Face Recognition ---
            elif self.mode == "type3":
                gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                faces_fmt = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in faces]
                draw_frame = face_render_node.draw(draw_frame, faces_fmt)

            # --- Type 4: Intrusion Detection ---
            elif self.mode == "type4":
                res = body_multi.detect(raw_frame)
                multi_data = [p['landmark_points'] for p in res.get('people', []) if 'landmark_points' in p]
                draw_frame = intrusion_renderer.draw(draw_frame, multi_data)

            # --- Type 5: Loitering Detection ---
            elif self.mode == "type5":
                res = body_multi.detect(raw_frame)
                multi_data = [p['landmark_points'] for p in res.get('people', []) if 'landmark_points' in p]
                draw_frame = loitering_renderer.draw(draw_frame, multi_data)

            # --- Type 6: Static Detection ---
            elif self.mode == "type6":
                res = body_multi.detect(raw_frame)
                multi_data = [p['landmark_points'] for p in res.get('people', []) if 'landmark_points' in p]
                draw_frame = static_renderer.draw(draw_frame, multi_data)

            # --- Type 7: Vigorous Activity ---
            elif self.mode == "type7":
                res = body_single.detect(raw_frame)
                pts = res.get('landmark_points') if res else None
                draw_frame = vigorous_renderer.draw(draw_frame, pts)

            # --- Type 8: Activity Level ---
            elif self.mode == "type8":
                res = body_single.detect(raw_frame)
                pts = res.get('landmark_points') if res else None
                draw_frame = activity_renderer.draw(draw_frame, pts)

            # --- Type 9: RAW ---
            elif self.mode == "type9":
                cv2.putText(draw_frame, "ENGINE STATUS: RAW_DATA_OK", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        except Exception as e:
            logger.error(f"Mode {self.mode} 算法执行报错: {e}")

        # 统一输出分辨率，确保 9 路矩阵显示一致
        return cv2.resize(draw_frame, (640, 480))


# --- 3. 信令与连接管理 (保持之前版本) ---
pcs = set()


async def offer(request):
    params = await request.json()
    offer_obj = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    mode = params.get("mode", "type1")
    shm_name = params.get("shm_name", "shiroha_frame")
    camera_id = params.get("camera_id", 0)

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_state():
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)
            await notify_flask_cleanup(camera_id)

    track = VideoProcessorTrack(mode=mode, shm_name=shm_name)
    pc.addTrack(track)
    await pc.setRemoteDescription(offer_obj)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


async def notify_flask_cleanup(camera_id):
    flask_url = f"http://127.0.0.1:8080/api/camera/cleanup_one/{camera_id}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(flask_url) as resp:
                logger.info(f"♻️ 通知 Flask 释放 Camera-{camera_id}: {resp.status}")
    except:
        pass


if __name__ == "__main__":
    app = web.Application()
    app.router.add_post("/offer", offer)
    logger.info("🚀 WebRTC AI 矩阵渲染中心已启动 (9-Renderer Mode)")
    web.run_app(app, port=8888)