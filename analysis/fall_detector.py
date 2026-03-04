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
                 alarm_hold_duration=10.0):  # 新增参数
        """
        初始化摔倒检测器。
        注意：此版本主要基于 SkeletonSolver 的输出 (body_yaw, angles) 进行判断，
        而非原始的 (x, y) 坐标。
        """
        self.state = self.STAND
        self.ground_start_time = None
        self.fall_confirmed = False
        self.fall_confirmed_at = None  # 记录摔倒确认的时间戳

        # 阈值配置
        self.ground_threshold_sec = ground_threshold_sec
        self.tilt_fall_threshold = tilt_fall_threshold
        self.support_stand_threshold = support_stand_threshold
        self.support_unbalance_threshold = support_unbalance_threshold
        self.tilt_unbalance_threshold = tilt_unbalance_threshold
        self.velocity_falling_threshold = velocity_falling_threshold
        self.velocity_ground_threshold = velocity_ground_threshold
        self.tilt_stand_threshold = tilt_stand_threshold  # 用于判断是否恢复站立

        # 报警持续时间
        self.alarm_hold_duration = alarm_hold_duration  # 摔倒确认后，FALL: TRUE 持续的时间

        # 用于计算 "虚拟速度" 的变量
        self.prev_body_yaw = None
        self.prev_time = None
        self.prev_vy = 0  # 上次计算出的 "虚拟速度"

        # 存储最后一次计算的特征
        self.last_features = {"vy": 0, "tilt": 0, "support": 1}

    def _calculate_features_from_solver_output(self, skeleton):
        """
        根据 SkeletonSolver 的输出字典计算特征。
        """
        features = {"vy": 0, "tilt": 0, "support": 1}

        if skeleton is None:
            self.last_features = features  # 更新存储的特征
            return features

        # 计算 Tilt (使用 body_yaw)
        # TODO： Tilt需要优化，对于摔倒与站起的检测有跳动，未摔倒的检测非常容易判别为摔倒
        body_yaw = skeleton.get('body_yaw')
        if body_yaw is not None:
            features["tilt"] = abs(body_yaw)
        else:
            self.last_features = features  # 更新存储的特征
            return features

        # 计算 Vy (基于 body_yaw 的变化率
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

        # 更新上一帧数据
        self.prev_body_yaw = current_body_yaw
        self.prev_time = current_time

        # 计算 Support (基于 body_yaw)
        if abs(features["tilt"]) > 80:
            features["support"] = 0.1
        elif abs(features["tilt"]) > 45:
            features["support"] = 0.5
        else:
            features["support"] = 1.0

        # 更新存储的特征
        self.last_features = features
        return features

    def get_last_features(self):
        """
        公共方法，用于获取最后一次计算的特征值。
        """
        return self.last_features

    def update(self, skeleton):
        """
        更新检测状态并返回摔倒标志。
        """
        # 调用内部方法计算特征，这会自动更新 self.last_features
        self._calculate_features_from_solver_output(skeleton)

        # 获取刚刚计算好的特征
        features = self.last_features

        vy = features.get("vy", 0)
        tilt = features.get("tilt", 0)
        support = features.get("support", 1)
        now = time.time()

        # 如果已经确认摔倒，并且还在持续时间内，直接返回 True
        if self.fall_confirmed_at is not None:
            if now - self.fall_confirmed_at < self.alarm_hold_duration:
                # print(f"Alarm hold active, returning True. Elapsed: {now - self.fall_confirmed_at:.2f}s") # Debug
                return True
            else:
                # 持续时间已过，重置所有相关状态
                self.fall_confirmed = False
                self.fall_confirmed_at = None
                self.state = self.STAND  # 强制回到初始状态
                self.ground_start_time = None
                # print("Alarm hold duration passed, reset all states.") # Debug

        # STAND
        if self.state == self.STAND:
            if support < self.support_unbalance_threshold and tilt > self.tilt_unbalance_threshold:
                self.state = self.UNBALANCE

        # UNBALANCE
        elif self.state == self.UNBALANCE:
            if abs(vy) > 30 and tilt > self.tilt_fall_threshold / 2:
                self.state = self.FALLING

        # FALLING
        elif self.state == self.FALLING:
            if abs(vy) < 10 and tilt > self.tilt_fall_threshold:
                self.state = self.GROUND
                self.ground_start_time = now
                self.fall_confirmed = False  # 准备再次确认

        # GROUND
        elif self.state == self.GROUND:
            if self.ground_start_time is not None:
                elapsed_time = now - self.ground_start_time
                # 条件1: 在地面时间足够长
                if elapsed_time >= self.ground_threshold_sec:
                    if not self.fall_confirmed:
                        self.fall_confirmed = True
                        self.fall_confirmed_at = now  # 记录确认时间
                        # print(f"[{now}] FALL CONFIRMED after {elapsed_time:.2f}s on ground! Will hold for {self.alarm_hold_duration}s.")
                    # return True # 不立即返回，让外部逻辑检查 alarm_hold_duration

                # 条件2: 即使时间未到，如果仍然高度倾斜且速度缓慢，也确认
                if tilt > self.tilt_fall_threshold and abs(vy) < 5:
                    if not self.fall_confirmed:
                        self.fall_confirmed = True
                        self.fall_confirmed_at = now  # 记录确认时间
                        # print(f"[{now}] FALL CONFIRMED due to sustained flat posture! Will hold for {self.alarm_hold_duration}s.")
                    # return True # 不立即返回，让外部逻辑检查 alarm_hold_duration

            # 条件3: 如果倾斜角减小（身体开始直立），回到STAND
            if tilt < self.tilt_stand_threshold:
                self.state = self.STAND
                self.ground_start_time = None
                # 如果此时还没有确认摔倒，只是恢复站立，则重置 fall_confirmed
                # 如果已经确认摔倒，fall_confirmed_at 会处理后续逻辑
                if not self.fall_confirmed:
                    self.fall_confirmed = False

        # 最终返回值判断
        if self.fall_confirmed_at is not None and (now - self.fall_confirmed_at < self.alarm_hold_duration):
            return True

        # 否则返回 False
        return False

    def get_state_name(self):
        """
        返回当前状态的字符串名称，便于外部显示。
        """
        names = {self.STAND: "STAND", self.UNBALANCE: "UNBALANCE", self.FALLING: "FALLING", self.GROUND: "GROUND"}
        return names.get(self.state, "UNKNOWN")