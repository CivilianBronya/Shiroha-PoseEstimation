# -*- coding: utf-8 -*-
import cv2
from capture.camera import Camera
from pose.body_pose import BodyPose
from face.head_pose import HeadPose
from rig.skeleton import SkeletonSolver
from output.json_out import JsonOutput
from render.single_stickman_renderer import SingleStickmanRenderer
from render.multi_stickman_renderer import MultiStickmanRenderer
from render.fall_detector_renderer import FallDetectorRenderer
from analysis.fall_detector import FallDetector

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
cam = Camera()
body_single = BodyPose(num_poses=1)
body_multi = BodyPose(num_poses=4)
head = HeadPose()
solver = SkeletonSolver(filter_alpha=0.7)
out = JsonOutput()
Single_renderer = SingleStickmanRenderer()

# 初始化摔倒检测器和其渲染器
fall_detector = FallDetector(ground_threshold_sec=4.5)
fall_detector_renderer = FallDetectorRenderer(fall_detector)

# 初始化多人姿态识别渲染器
pose_recognition_renderer = MultiStickmanRenderer()

while True:
    frame = cam.read()
    if frame is None:
        continue

    # 获取 BodyPose 的原始数据（单人）
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
            head_rot = head.detect(frame)  # HeadPose 也可能需要当前帧
            skeleton = solver.solve(body_pts_dict, head_rot)
            out.send(skeleton)  # 发送单人数据

    # MotionCapture 窗口
    cam.show(frame)

    # FallDetector 窗口
    debug_frame_fall_detector = frame.copy()
    if body_pts_list is not None:
        debug_frame_fall_detector = Single_renderer.draw(debug_frame_fall_detector, body_pts_list, skeleton)
    debug_frame_fall_detector = fall_detector_renderer.draw(debug_frame_fall_detector, skeleton)
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

    # cv2.imshow(WINDOW_NAME_MAP["FaceRecognition"], frame)
    # cv2.imshow(WINDOW_NAME_MAP["IntrusionDetection"], frame)
    # cv2.imshow(WINDOW_NAME_MAP["EmotionDetection"], frame)
    # cv2.imshow(WINDOW_NAME_MAP["LoiteringDetection"], frame)
    # cv2.imshow(WINDOW_NAME_MAP["StaticDetection"], frame)
    # cv2.imshow(WINDOW_NAME_MAP["VigorousActivity"], frame)
    # cv2.imshow(WINDOW_NAME_MAP["ActivityLevel"], frame)

    if cv2.waitKey(1) == 27:  # ESC 键退出
        break

cv2.destroyAllWindows()