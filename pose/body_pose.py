import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import math

# Mediapipe新版
class BodyPose:
    def __init__(self):
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="models/pose_landmarker_full.task"),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.timestamp = 0

    def _calculate_raw_yaw_from_world_landmarks(self, world_landmarks):
        """
        基于3D世界坐标计算原始的 body_yaw 角度 (范围 [-180, 180] 度)。
        使用左肩和右肩的中点以及左髋和右髋的中点来估算身体朝向。
        """
        if len(world_landmarks) < 24: # 确保有足够的关键点
            return None

        # 获取关键点索引 (参考 MediaPipe pose landmarks 定义)
        LEFT_SHOULDER_IDX = 11
        RIGHT_SHOULDER_IDX = 12
        LEFT_HIP_IDX = 23
        RIGHT_HIP_IDX = 24

        try:
            left_shoulder = world_landmarks[LEFT_SHOULDER_IDX]
            right_shoulder = world_landmarks[RIGHT_SHOULDER_IDX]
            left_hip = world_landmarks[LEFT_HIP_IDX]
            right_hip = world_landmarks[RIGHT_HIP_IDX]

            # 计算肩部中点和髋部中点
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2.0
            shoulder_center_z = (left_shoulder.z + right_shoulder.z) / 2.0 # Z 表示深度，X-Z平面构成水平面
            hip_center_x = (left_hip.x + right_hip.x) / 2.0
            hip_center_z = (left_hip.z + right_hip.z) / 2.0

            # 计算从髋部中心指向肩部中心的向量 (在 X-Z 平面上)
            dx = shoulder_center_x - hip_center_x
            dz = shoulder_center_z - hip_center_z

            # 计算该向量与 X 轴正方向的夹角 (atan2 返回 [-pi, pi])
            raw_yaw_rad = math.atan2(dz, dx) # 注意： atan2(y, x)，这里 Z 是 Y，X 是 X

            # 转换为度数
            raw_yaw_deg = math.degrees(raw_yaw_rad)

            # 将角度映射到 [-180, 180]
            while raw_yaw_deg > 180:
                raw_yaw_deg -= 360
            while raw_yaw_deg <= -180:
                raw_yaw_deg += 360

            return raw_yaw_deg
        except (IndexError, AttributeError):
            # 如果关键点不存在或访问出错
            return None


    def detect(self, frame):
        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.timestamp += 23  # 约30FPS时间推进(ms)

        result = self.detector.detect_for_video(mp_image, self.timestamp)

        if not result.pose_landmarks:
            return None

        h, w, _ = frame.shape
        pts = []

        for lm in result.pose_landmarks[0]:
            pts.append((lm.x*w, lm.y*h, lm.z))

        # 计算原始 body_yaw (仅使用world_landmarks)
        raw_body_yaw_deg = None
        if result.pose_world_landmarks and result.pose_world_landmarks[0]:
             raw_body_yaw_deg = self._calculate_raw_yaw_from_world_landmarks(result.pose_world_landmarks[0])

        # print(f"Raw Yaw: {raw_body_yaw_deg}") # Debug line

        # 返回关键点和原始yaw
        return {
            'landmark_points': pts,
            'raw_body_yaw': raw_body_yaw_deg # 仅返回原始值
        }

# 示例
# cap = cv2.VideoCapture(0) # 打开摄像头
# bp = BodyPose() # 创建实例

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     result_data = bp.detect(frame)
#     if result_data:
#         raw_yaw = result_data['raw_body_yaw']
#         if raw_yaw is not None:
#             print(f"Raw Yaw: {raw_yaw:.2f} deg")
#         # ... 其他处理 ...

#     cv2.imshow('Camera Feed', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()