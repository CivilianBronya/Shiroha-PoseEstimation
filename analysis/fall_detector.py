# analysis/fall_detector.py
import time
import cv2
import math


class FallDetector:
    # 状态定义
    STAND = 0
    UNBALANCE = 1
    FALLING = 2
    GROUND = 3

    def __init__(self, ground_threshold_sec=4.5, tilt_fall_threshold=60,
                 support_stand_threshold=0.7, support_unbalance_threshold=0.4,
                 tilt_unbalance_threshold=20, velocity_falling_threshold=15,
                 velocity_ground_threshold=2, tilt_stand_threshold=25,
                 alarm_hold_duration=10.0):
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

        self.prev_body_yaw = None
        self.prev_time = None
        self.prev_vy = 0

        self.last_features = {"vy": 0, "tilt": 0, "support": 1}

        # 保留摔倒状态判断逻辑
        self.is_falling = False


    def _calculate_features_from_solver_output(self, skeleton):
        features = {"vy": 0, "tilt": 0, "support": 1}

        if skeleton is None:
            self.last_features = features
            return features

        body_yaw = skeleton.get('body_yaw')
        if body_yaw is not None:
            features["tilt"] = abs(body_yaw)
        else:
            self.last_features = features
            return features

        current_body_yaw = body_yaw
        current_time = time.time()

        if self.prev_body_yaw is not None and self.prev_time is not None:
            d_yaw = current_body_yaw - self.prev_body_yaw
            dt = current_time - self.prev_time
            if dt > 0:
                virtual_velocity = d_yaw / dt
                features["vy"] = virtual_velocity
                self.prev_vy = virtual_velocity
            else:
                features["vy"] = self.prev_vy
        else:
            features["vy"] = self.prev_vy

        self.prev_body_yaw = current_body_yaw
        self.prev_time = current_time

        if abs(features["tilt"]) > 80:
            features["support"] = 0.1
        elif abs(features["tilt"]) > 45:
            features["support"] = 0.5
        else:
            features["support"] = 1.0

        self.last_features = features
        return features

    def get_last_features(self):
        return self.last_features

    def update(self, skeleton):
        """
        更新检测状态。
        注意：保留了所有 is_falling 的判断逻辑，但不返回 True/False。
        """
        self._calculate_features_from_solver_output(skeleton)
        features = self.last_features

        vy = features.get("vy", 0)
        tilt = features.get("tilt", 0)
        support = features.get("support", 1)
        now = time.time()

        if self.fall_confirmed_at is not None:
            if now - self.fall_confirmed_at < self.alarm_hold_duration:
                self.is_falling = True
                return
            else:
                self.fall_confirmed = False
                self.fall_confirmed_at = None
                self.state = self.STAND
                self.ground_start_time = None

        if self.state == self.STAND:
            if support < self.support_unbalance_threshold and tilt > self.tilt_unbalance_threshold:
                self.state = self.UNBALANCE

        elif self.state == self.UNBALANCE:
            if abs(vy) > 30 and tilt > self.tilt_fall_threshold / 2:
                self.state = self.FALLING

        elif self.state == self.FALLING:
            if abs(vy) < 10 and tilt > self.tilt_fall_threshold:
                self.state = self.GROUND
                self.ground_start_time = now
                self.fall_confirmed = False

        elif self.state == self.GROUND:
            if self.ground_start_time is not None:
                elapsed_time = now - self.ground_start_time

                if elapsed_time >= self.ground_threshold_sec:
                    if not self.fall_confirmed:
                        self.fall_confirmed = True
                        self.fall_confirmed_at = now

                if tilt > self.tilt_fall_threshold and abs(vy) < 5:
                    if not self.fall_confirmed:
                        self.fall_confirmed = True
                        self.fall_confirmed_at = now

            if tilt < self.tilt_stand_threshold:
                self.state = self.STAND
                self.ground_start_time = None
                if not self.fall_confirmed:
                    self.fall_confirmed = False

        # 保留的 is_falling 判断逻辑
        if self.fall_confirmed_at is not None and (now - self.fall_confirmed_at < self.alarm_hold_duration):
            self.is_falling = True
        else:
            self.is_falling = False

    def get_state_name(self):
        names = {self.STAND: "STAND", self.UNBALANCE: "UNBALANCE", self.FALLING: "FALLING", self.GROUND: "GROUND"}
        return names.get(self.state, "UNKNOWN")

    # 保留的 is_falling 查询方法
    def get_fall_status(self):
        """
        获取当前的摔倒判断结果 (True/False)。
        """
        return self.is_falling

    # 返回一个数值型的风险阈值
    def get_fall_risk_score(self):
        """
        返回一个 0-100 的数值，表示当前的摔倒风险等级。
        这个分数可以由前端根据需要自行判断。
        """
        features = self.last_features
        tilt = features.get("tilt", 0)
        vy = features.get("vy", 0)
        support = features.get("support", 1)

        score = 0

        if tilt > self.tilt_fall_threshold:
            score += 60
        elif tilt > self.tilt_unbalance_threshold:
            # 使用一个平滑的曲线，例如线性插值
            score += 60 * (tilt - self.tilt_unbalance_threshold) / (
                        self.tilt_fall_threshold - self.tilt_unbalance_threshold)

        score += (1 - support) * 20

        if self.state == self.FALLING or self.state == self.GROUND:
            if abs(vy) < self.velocity_ground_threshold:
                score += 20
            elif abs(vy) > self.velocity_falling_threshold:
                score += 10

        final_score = max(0, min(100, score))

        self.last_features['score'] = final_score

        return final_score