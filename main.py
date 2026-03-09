# -*- coding: utf-8 -*-
import cv2
import json
import threading
import time
import asyncio
import websockets
import numpy as np
from capture.camera import Camera
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

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# 定义窗口名映射
WINDOW_NAME_MAP = {
    "FallDetector": "Fall Detector",
    "PoseMonitoring": "Pose Monitoring",
    "MotionCapture": "Motion Capture",
    "FaceRecognition": "Face Recognition",
    "IntrusionDetection": "Intrusion Detection",
    "EmotionDetection": "Emotion Detection",
    "LoiteringDetection": "Loitering Detection",
    "StaticDetection": "Static Detection",
    "VigorousActivity": "Vigorous Activity",
    "ActivityLevel": "Activity Level"
}

# 初始化组件
cam = Camera(
    index=config["camera"]["index"],
    width=config["camera"]["width"],
    height=config["camera"]["height"],
    fps=config["camera"]["fps"]
)

body_single = BodyPose(num_poses=config["pose"]["single"]["num_poses"])
body_multi = BodyPose(num_poses=config["pose"]["multi"]["num_poses"])
head = HeadPose()
solver = SkeletonSolver(filter_alpha=config["skeleton"]["filter_alpha"])
Single_renderer = SingleStickmanRenderer()

# 使用配置初始化渲染器
intrusion_detection_renderer = IntrusionDetectionRenderer()
loitering_detection_renderer = LoiteringDetectionRenderer(
    alert_duration=config["detection"]["loitering"]["alert_duration"],
    cycle_length=config["detection"]["loitering"]["cycle_length"],
    alert_threshold=config["detection"]["loitering"]["alert_threshold"]
)
static_detection_renderer = StaticDetectionRenderer(
    history_length=config["detection"]["static"]["history_length"],
    movement_threshold=config["detection"]["static"]["movement_threshold"]
)
vigorous_activity_renderer = VigorousActivityRenderer(
    activity_threshold=config["detection"]["vigorous_activity"]["activity_threshold"]
)
activity_level_renderer = ActivityLevelRenderer(
    low_threshold=config["detection"]["activity_level"]["low_threshold"],
    high_threshold=config["detection"]["activity_level"]["high_threshold"]
)

face_solver_multi = FaceSolver(
    filter_alpha=config["face"]["filter_alpha"],
    min_area=config["face"]["min_area"],
    mode=MODE_MULTI
)
face_recognition_renderer_multi = FaceRecognitionRenderer(face_solver_multi)

face_solver_single = FaceSolver(
    filter_alpha=config["face"]["filter_alpha"],
    min_area=config["face"]["min_area"],
    mode=MODE_SINGLE
)
face_recognition_renderer_single = FaceRecognitionRenderer(face_solver_single)

# 初始化渲染器
fall_detector = FallDetector(ground_threshold_sec=config["fall_detector"]["ground_threshold_sec"])
fall_detector_renderer = FallDetectorRenderer(fall_detector)
pose_recognition_renderer = MultiStickmanRenderer()

latest_processed_frame = None
frame_lock = threading.Lock()

WEBSOCKET_URI = "ws://localhost:8765"
WEBSOCKET_CLIENT = None
CLIENT_SESSION_ID = None


async def run_websocket_client():
    """
    异步函数，处理WebSocket的连接、认证和数据发送。
    """
    global WEBSOCKET_CLIENT, CLIENT_SESSION_ID, latest_processed_frame
    uri = WEBSOCKET_URI

    print(f"Attempting to connect to WebSocket server at {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            WEBSOCKET_CLIENT = websocket

            # 等待服务器的欢迎消息以获取Session ID
            welcome_response = await websocket.recv()
            welcome_data = json.loads(welcome_response)

            if welcome_data.get("type") == "welcome":
                CLIENT_SESSION_ID = welcome_data.get("session_id")
                print(f"WebSocket Connected. Assigned Session ID: {CLIENT_SESSION_ID}")
            else:
                print("Unexpected welcome message from server.")
                return

            # 循环发送帧
            while True:
                with frame_lock:
                    frame_to_send = latest_processed_frame.copy() if latest_processed_frame is not None else None

                if frame_to_send is not None:
                    try:
                        success, encoded_image = cv2.imencode('.jpg', frame_to_send,
                                                              [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        if not success:
                            print("Failed to encode image for streaming via WebSocket.")
                            await asyncio.sleep(0.03)
                            continue

                        encoded_bytes = encoded_image.tobytes()

                        # 发送元数据 (JSON)
                        metadata = {
                            "type": "video_frame",
                            "session_id": CLIENT_SESSION_ID,
                            "timestamp": time.time()
                        }
                        await websocket.send(json.dumps(metadata))

                        # 发送二进制图像数据
                        await websocket.send(encoded_bytes)

                    except websockets.exceptions.ConnectionClosedOK:
                        print("WebSocket connection closed by server (cleanly).")
                        break
                    except websockets.exceptions.ConnectionClosedError as e:
                        print(f"WebSocket connection closed by server with error: {e}")
                        break
                    except Exception as e:
                        print(f"Error sending frame via WebSocket: {e}")
                        # 可能需要重连逻辑，此处简单break
                        break

                # 等待一段时间再发送下一帧
                await asyncio.sleep(0.03)  # 约33 FPS

    except OSError as e:
        print(f"Could not connect to WebSocket server at {uri}. Is it running? Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in WebSocket client: {e}")
    finally:
        WEBSOCKET_CLIENT = None
        CLIENT_SESSION_ID = None
        print("WebSocket client disconnected.")


def stream_thread():
    """
    一个独立的线程，启动一个异步事件循环来处理 WebSocket 连接和发送。
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_websocket_client())


# 启动流式传输线程
streamer = threading.Thread(target=stream_thread, daemon=True)
streamer.start()

try:
    while True:
        frame, dt = cam.read()
        if frame is None:
            continue

        # 获取 BodyPose 的原始数据
        raw_body_result_single = body_single.detect(frame)
        body_pts_list = None
        raw_body_yaw = None
        skeleton = None

        if raw_body_result_single is not None and isinstance(raw_body_result_single, dict):
            body_pts_list = raw_body_result_single.get('landmark_points')
            raw_body_yaw = raw_body_result_single.get('raw_body_yaw')

            # 解算单人骨架
            if body_pts_list is not None:
                body_pts_dict = {i: pt for i, pt in enumerate(body_pts_list)}
                head_rot = head.detect(frame)
                skeleton = solver.solve(body_pts_dict, head_rot)
                # 移除：out.send(skeleton) 调用

        # MotionCapture 窗口
        frame_with_fps = frame.copy()
        if dt > 0:
            fps = int(1 / dt)
            cv2.putText(frame_with_fps, f"FPS:{fps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME_MAP["MotionCapture"], frame_with_fps)

        # FallDetector 窗口
        debug_frame_fall_detector = frame.copy()
        if body_pts_list is not None:
            debug_frame_fall_detector = Single_renderer.draw(debug_frame_fall_detector, body_pts_list, skeleton)
        debug_frame_fall_detector = fall_detector_renderer.draw(debug_frame_fall_detector, skeleton, body_pts_list)
        cv2.imshow(WINDOW_NAME_MAP["FallDetector"], debug_frame_fall_detector)

        # PoseMonitoring 多人姿态
        debug_frame_pose_monitoring = frame.copy()
        raw_body_result_multi = body_multi.detect(frame)
        multi_body_data = []
        if raw_body_result_multi is not None and 'people' in raw_body_result_multi:
            for person in raw_body_result_multi['people']:
                if 'landmark_points' in person:
                    multi_body_data.append(person['landmark_points'])

        # 使用 MultiStickmanRenderer 渲染多人
        if multi_body_data:
            debug_frame_pose_monitoring = pose_recognition_renderer.draw(debug_frame_pose_monitoring, multi_body_data)
        cv2.imshow(WINDOW_NAME_MAP["PoseMonitoring"], debug_frame_pose_monitoring)

        # FaceRecognition 窗口
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_raw = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 将检测结果转换为渲染器期望的格式 [[x, y, w, h], ...]
        faces_for_render = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in faces_raw]
        debug_frame_face_recognition = frame.copy()
        debug_frame_face_recognition = face_recognition_renderer_multi.draw(debug_frame_face_recognition,
                                                                            faces_for_render)
        cv2.imshow(WINDOW_NAME_MAP["FaceRecognition"], debug_frame_face_recognition)

        # EmotionDetection 窗口
        debug_frame_emotion = frame.copy()
        debug_frame_emotion = face_recognition_renderer_single.draw(debug_frame_emotion, faces_for_render)
        cv2.imshow(WINDOW_NAME_MAP["EmotionDetection"], debug_frame_emotion)

        # IntrusionDetection 窗口
        debug_frame_intrusion = frame.copy()
        debug_frame_intrusion = intrusion_detection_renderer.draw(debug_frame_intrusion, multi_body_data)
        cv2.imshow(WINDOW_NAME_MAP["IntrusionDetection"], debug_frame_intrusion)

        # LoiteringDetection 窗口
        debug_frame_loitering = frame.copy()
        debug_frame_loitering = loitering_detection_renderer.draw(debug_frame_loitering,
                                                                  multi_body_data)  # 传入多人数据（即使没用到）
        cv2.imshow(WINDOW_NAME_MAP["LoiteringDetection"], debug_frame_loitering)

        # StaticDetection 窗口
        debug_frame_static = frame.copy()
        debug_frame_static = static_detection_renderer.draw(debug_frame_static, multi_body_data)
        cv2.imshow(WINDOW_NAME_MAP["StaticDetection"], debug_frame_static)

        debug_frame_vigorous = frame.copy()
        debug_frame_vigorous = vigorous_activity_renderer.draw(debug_frame_vigorous, body_pts_list)
        cv2.imshow(WINDOW_NAME_MAP["VigorousActivity"], debug_frame_vigorous)

        debug_frame_activity = frame.copy()
        debug_frame_activity = activity_level_renderer.draw(debug_frame_activity, body_pts_list)
        cv2.imshow(WINDOW_NAME_MAP["ActivityLevel"], debug_frame_activity)

        with frame_lock:
            latest_processed_frame = debug_frame_fall_detector.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 键退出
            print("ESC pressed, exiting...")
            break

except KeyboardInterrupt:
    print("\n程序被用户中断")

finally:
    # 释放资源
    print("Releasing camera resources...")
    cam.release()
    print("Closing all windows...")
    cv2.destroyAllWindows()

print("Program exited cleanly.")