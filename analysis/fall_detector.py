import math
import time


class FallDetector:
    # 状态定义
    STAND = 0
    UNBALANCE = 1
    FALLING = 2
    GROUND = 3

    def __init__(self, ground_threshold_sec=3.0, tilt_fall_threshold=50,
                 support_stand_threshold=0.7, support_unbalance_threshold=0.4,
                 tilt_unbalance_threshold=25, velocity_falling_threshold=50,
                 velocity_ground_threshold=10, tilt_stand_threshold=20,
                 alarm_hold_duration=10.0, smoothing_factor=0.3):
        """
        初始化摔倒检测器。
        """
        self.state = self.STAND
        self.ground_start_time = None
        self.fall_confirmed = False
        self.fall_confirmed_at = None

        # 阈值配置
        self.ground_threshold_sec = ground_threshold_sec
        self.tilt_fall_threshold = tilt_fall_threshold
        self.support_stand_threshold = support_stand_threshold
        self.support_unbalance_threshold = support_unbalance_threshold
        self.tilt_unbalance_threshold = tilt_unbalance_threshold
        self.velocity_falling_threshold = velocity_falling_threshold
        self.velocity_ground_threshold = velocity_ground_threshold
        self.tilt_stand_threshold = tilt_stand_threshold
        self.alarm_hold_duration = alarm_hold_duration

        self.smoothing_factor = smoothing_factor

        self.prev_centroid_y = None
        self.prev_time = None
        self.smoothed_vy = 0

        self.last_features = {"vy": 0, "tilt": 0, "support": 1}
        self.last_update_time = time.time()
        # 保留摔倒状态判断逻辑
        self.is_falling = False

    def _calculate_features_from_keypoints(self, skeleton, keypoints_2d):
        """
        从骨架数据和原始关键点计算特征。
        skeleton: SkeletonSolver.solve() 的返回值
        keypoints_2d: 原始的2D关键点列表 [(x,y,c), ...]
        """
        features = {"vy": 0, "tilt": 0, "support": 1}

        # 如果输入数据无效，立即返回一个“不可信”的默认状态
        if skeleton is None or keypoints_2d is None or len(keypoints_2d) < 7:
            features["support"] = 0.0
            features["vy"] = self.smoothed_vy
            features["tilt"] = 0.0
            self.last_features = features
            return features

        # 计算 TILT (俯仰角 - Pitch)
        NECK_IDX = 0
        LEFT_SHOULDER_IDX = 1
        RIGHT_SHOULDER_IDX = 2
        LEFT_HIP_IDX = 5
        RIGHT_HIP_IDX = 6

        # 获取关键点坐标
        neck_point = keypoints_2d[NECK_IDX] if NECK_IDX < len(keypoints_2d) else None
        left_shoulder_point = keypoints_2d[LEFT_SHOULDER_IDX] if LEFT_SHOULDER_IDX < len(keypoints_2d) else None
        right_shoulder_point = keypoints_2d[RIGHT_SHOULDER_IDX] if RIGHT_SHOULDER_IDX < len(keypoints_2d) else None
        left_hip_point = keypoints_2d[LEFT_HIP_IDX] if LEFT_HIP_IDX < len(keypoints_2d) else None
        right_hip_point = keypoints_2d[RIGHT_HIP_IDX] if RIGHT_HIP_IDX < len(keypoints_2d) else None

        required_points = [neck_point, left_shoulder_point, right_shoulder_point, left_hip_point, right_hip_point]
        if not all(p is not None and p[2] > 0.1 for p in required_points):
            features["support"] = 0.05
            features["vy"] = self.smoothed_vy
            features["tilt"] = self.last_features.get("tilt", 0)
            self.last_features = features
            return features

        # 计算肩部中心点
        shoulder_center_x = (left_shoulder_point[0] + right_shoulder_point[0]) / 2
        shoulder_center_y = (left_shoulder_point[1] + right_shoulder_point[1]) / 2

        # 计算髋部中心点
        hip_center_x = (left_hip_point[0] + right_hip_point[0]) / 2
        hip_center_y = (left_hip_point[1] + right_hip_point[1]) / 2

        # 计算肩-髋连线与水平线的夹角 (俯仰角)
        dx = hip_center_x - shoulder_center_x
        dy = hip_center_y - shoulder_center_y
        if dx != 0:
            pitch_rad = math.atan2(dy, dx)
            pitch_deg = math.degrees(pitch_rad)
            features["tilt"] = abs(pitch_deg)
        else:
            # 如果dx为0，说明肩髋在同一垂线上，可以认为是极端倾斜
            features["tilt"] = 90

        # 2. 计算 SUPPORT (支撑因子) ---
        tilt = features["tilt"]
        if tilt > 80:
            features["support"] = 0.1
        elif tilt > 45:
            features["support"] = 0.5
        else:
            features["support"] = 1.0

        # 计算 VY (真实垂直速度)
        total_y = 0
        valid_points = 0
        for idx in [NECK_IDX, LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX, LEFT_HIP_IDX, RIGHT_HIP_IDX]:
            if idx < len(keypoints_2d):
                point = keypoints_2d[idx]
                if point[2] > 0.1:
                    total_y += point[1]
                    valid_points += 1

        if valid_points > 0:
            current_centroid_y = total_y / valid_points
            current_time = time.time()

            if self.prev_centroid_y is not None and self.prev_time is not None:
                dt = current_time - self.prev_time
                if dt > 0:
                    raw_vy = (self.prev_centroid_y - current_centroid_y) / dt

                    self.smoothed_vy = (self.smoothing_factor * raw_vy +
                                        (1 - self.smoothing_factor) * self.smoothed_vy)

                    features["vy"] = self.smoothed_vy
                else:
                    features["vy"] = self.smoothed_vy
            else:
                features["vy"] = 0

            self.prev_centroid_y = current_centroid_y
            self.prev_time = current_time
        else:
            features["vy"] = self.smoothed_vy

        # 更新最后的特征并返回
        self.last_features = features
        return features

    def get_last_features(self):
        return self.last_features

    def update(self, skeleton, keypoints_2d=None):
        """
        更新检测状态。
        :param skeleton: SkeletonSolver.solve() 的返回值。
        :param keypoints_2d: 原始的2D关键点列表。
        """
        # 每次 update 都强制计算新特征，即使关键点丢失
        self._calculate_features_from_keypoints(skeleton, keypoints_2d)
        features = self.last_features

        vy = features.get("vy", 0)
        tilt = features.get("tilt", 0)
        support = features.get("support", 1)
        now = time.time()

        # 长时间没有更新，重置状态
        if now - self.last_update_time > 1.0:  # 超过1秒未更新
            self.reset_to_stand()
            return

        # 更新最后更新时间
        self.last_update_time = now

        if self.fall_confirmed_at is not None:
            if now - self.fall_confirmed_at < self.alarm_hold_duration:
                self.is_falling = True
                return
            else:
                self.reset_to_stand()
                return

        if self.state == self.STAND:
            if support < self.support_unbalance_threshold and tilt > self.tilt_unbalance_threshold:
                self.state = self.UNBALANCE

        elif self.state == self.UNBALANCE:
            if abs(vy) > self.velocity_falling_threshold / 2 and tilt > self.tilt_fall_threshold / 2:
                self.state = self.FALLING

        elif self.state == self.FALLING:
            if abs(vy) < self.velocity_ground_threshold and tilt > self.tilt_fall_threshold:
                self.state = self.GROUND
                self.ground_start_time = now

        elif self.state == self.GROUND:
            if self.ground_start_time is not None:
                elapsed_time = now - self.ground_start_time

                # 满足时长或满足高倾角+低速条件
                if elapsed_time >= self.ground_threshold_sec or \
                        (tilt > self.tilt_fall_threshold and abs(vy) < self.velocity_ground_threshold / 2):
                    if not self.fall_confirmed:
                        self.fall_confirmed = True
                        self.fall_confirmed_at = now

                # 倾斜角变小，正在起身
                if tilt < self.tilt_stand_threshold:
                    self.state = self.STAND
                    self.ground_start_time = None
                    if not self.fall_confirmed:
                        self.fall_confirmed = False

        self.is_falling = self.fall_confirmed

    def reset_to_stand(self):
        """重置所有状态为站立"""
        self.state = self.STAND
        self.ground_start_time = None
        self.fall_confirmed = False
        self.fall_confirmed_at = None
        self.is_falling = False
        self.prev_centroid_y = None
        self.prev_time = None
        self.smoothed_vy = 0
        self.last_features = {"vy": 0, "tilt": 0, "support": 1}

    def get_state_name(self):
        names = {self.STAND: "STAND", self.UNBALANCE: "UNBALANCE", self.FALLING: "FALLING", self.GROUND: "GROUND"}
        return names.get(self.state, "UNKNOWN")

    def get_fall_status(self):
        """
        获取当前的摔倒判断结果 (True/False)。
        """
        return self.is_falling

    def get_fall_risk_score(self):
        """
        返回一个 0-100 的数值，表示当前的摔倒风险等级。
        此版本将状态机的判断结果也融入评分中，并加入置信度标记。
        """
        features = self.last_features
        tilt = features.get("tilt", 0)
        vy = features.get("vy", 0)
        support = features.get("support", 1)

        # 置信度评估
        is_data_reliable = support > 0.05

        score = 0

        # 基于倾斜角的风险
        if tilt > self.tilt_fall_threshold:
            score += 60
        elif tilt > self.tilt_unbalance_threshold:
            score += 60 * (tilt - self.tilt_unbalance_threshold) / (
                    self.tilt_fall_threshold - self.tilt_unbalance_threshold)

        # 基于支撑的风险
        score += (1 - support) * 20

        # 基于速度的风险
        if abs(vy) > self.velocity_falling_threshold:
            score += 15
        elif abs(vy) < self.velocity_ground_threshold and tilt > self.tilt_unbalance_threshold:
            score += 10

        if self.fall_confirmed:
            score = 100

        # 基于状态的风险加权
        state_bonus = {
            self.UNBALANCE: 10,
            self.FALLING: 30,
            self.GROUND: 50
        }
        score += state_bonus.get(self.state, 0)

        final_score = max(0, min(100, score))

        if is_data_reliable:
            result = str(int(final_score))
        else:
            # 数据不可靠时，在分数后加上 ??
            result = f"{int(final_score)}??"

        self.last_features['score'] = result
        return result