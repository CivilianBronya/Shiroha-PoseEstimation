import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import math

class BodyPose:
    def __init__(self, model_path="models/pose_landmarker_full.task", num_poses=1):
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.timestamp = 0
        self.is_multi_mode = num_poses > 1

    def _calculate_raw_yaw_from_world_landmarks(self, world_landmarks):
        """
        基于3D世界坐标计算原始的 body_yaw 角度 (范围 [-180, 180] 度)。
        使用左肩和右肩的中点以及左髋和右髋的中点来估算身体朝向。
        """
        if len(world_landmarks) < 24:  # 确保有足够的关键点
            return None

        LEFT_SHOULDER_IDX = 11
        RIGHT_SHOULDER_IDX = 12
        LEFT_HIP_IDX = 23
        RIGHT_HIP_IDX = 24

        try:
            left_shoulder = world_landmarks[LEFT_SHOULDER_IDX]
            right_shoulder = world_landmarks[RIGHT_SHOULDER_IDX]
            left_hip = world_landmarks[LEFT_HIP_IDX]
            right_hip = world_landmarks[RIGHT_HIP_IDX]

            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2.0
            shoulder_center_z = (left_shoulder.z + right_shoulder.z) / 2.0
            hip_center_x = (left_hip.x + right_hip.x) / 2.0
            hip_center_z = (left_hip.z + right_hip.z) / 2.0

            dx = shoulder_center_x - hip_center_x
            dz = shoulder_center_z - hip_center_z

            raw_yaw_rad = math.atan2(dz, dx)
            raw_yaw_deg = math.degrees(raw_yaw_rad)

            while raw_yaw_deg > 180:
                raw_yaw_deg -= 360
            while raw_yaw_deg <= -180:
                raw_yaw_deg += 360

            return raw_yaw_deg
        except (IndexError, AttributeError):
            return None

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.timestamp += 10

        result = self.detector.detect_for_video(mp_image, self.timestamp)

        people_data = []
        if result.pose_landmarks:
            for idx, pose_landmarks in enumerate(result.pose_landmarks):
                pts = []
                for lm in pose_landmarks:
                    pts.append((lm.x * w, lm.y * h, lm.z))

                raw_body_yaw_deg = None
                if result.pose_world_landmarks and idx < len(result.pose_world_landmarks):
                    raw_body_yaw_deg = self._calculate_raw_yaw_from_world_landmarks(result.pose_world_landmarks[idx])

                people_data.append({
                    'landmark_points': pts,
                    'raw_body_yaw': raw_body_yaw_deg
                })

        if self.is_multi_mode:
            # 多人模式：返回包含 people 列表的字典
            return {'people': people_data} if people_data else None
        else:
            # 单人模式：为了向后兼容，返回第一个人的数据（如果存在）
            if people_data:
                return people_data[0]
            else:
                return None

    def detect_multi(self, frame):
        """专门用于多人检测的接口"""
        original_is_multi = self.is_multi_mode
        self.is_multi_mode = True
        result = self.detect(frame)
        self.is_multi_mode = original_is_multi
        return result

    def detect_single(self, frame):
        """专门用于单人检测的接口"""
        original_is_multi = self.is_multi_mode
        self.is_multi_mode = False
        result = self.detect(frame)
        self.is_multi_mode = original_is_multi
        return result