import math
from smooth.filter import LowPassFilter


def vec(a, b):
    return (a[0] - b[0], a[1] - b[1])


def angle3(a, b, c):
    # 以 b 为关节
    ab = vec(a, b)
    cb = vec(c, b)

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    lab = math.hypot(*ab)
    lcb = math.hypot(*cb)

    if lab * lcb == 0:
        return 0

    cos = max(-1, min(1, dot / (lab * lcb)))
    return math.degrees(math.acos(cos))


class SkeletonSolver:
    def __init__(self, filter_alpha=0.7):
        self.filter_alpha = filter_alpha
        # 初始化用于角度解缠绕和平滑的变量
        self._previous_raw_angle_rad = None
        self._unwrapped_angle_rad = 0.0  # 存储连续的解缠绕角度
        self._yaw_filter = LowPassFilter(alpha=self.filter_alpha)

    def solve(self, body, head):
        if body is None:
            return None

        sk = {}

        # 手肘（肩-肘-腕）
        sk["left_elbow"] = angle3(body[11], body[13], body[15])
        sk["right_elbow"] = angle3(body[12], body[14], body[16])

        # 膝盖（髋-膝-踝）
        sk["left_knee"] = angle3(body[23], body[25], body[27])
        sk["right_knee"] = angle3(body[24], body[26], body[28])

        # 身体朝向（肩线方向）- 原始角度计算
        shoulder = vec(body[12], body[11])
        raw_yaw_rad = math.atan2(shoulder[1], shoulder[0])
        raw_yaw_deg = math.degrees(raw_yaw_rad)

        # 将原始角度映射到 [-180, 180] 范围（虽然 atan2 保证了这一点，但保险起见）
        raw_yaw_deg = ((raw_yaw_deg + 180) % 360) - 180

        # 解缠绕 (Unwrap)
        if self._previous_raw_angle_rad is not None:
            # 计算当前角度与上一角度的差值
            delta_angle = raw_yaw_rad - self._previous_raw_angle_rad
            # 将差值规范化到 [-π, π] 范围内
            while delta_angle > math.pi:
                delta_angle -= 2 * math.pi
            while delta_angle <= -math.pi:
                delta_angle += 2 * math.pi

            # 更新连续的解缠绕角度
            self._unwrapped_angle_rad += delta_angle
        else:
            # 直接使用原始角度作为解缠绕的起点
            self._unwrapped_angle_rad = raw_yaw_rad

        # 更新_previous_raw_angle_rad 为本次的原始角度
        self._previous_raw_angle_rad = raw_yaw_rad

        unwrapped_deg = math.degrees(self._unwrapped_angle_rad)
        print(f"SkeletonSolver Internal - Unwrapped body_yaw: {unwrapped_deg:.3f}")

        # 将解缠绕后的弧度送入滤波器
        filtered_unwrapped_rad = self._yaw_filter.apply(self._unwrapped_angle_rad)

        # 将滤波后的解缠绕角度重新映射回 [-π, π] 范围
        temp_filtered_rad = filtered_unwrapped_rad
        while temp_filtered_rad > math.pi:
            temp_filtered_rad -= 2 * math.pi
        while temp_filtered_rad <= -math.pi:
            temp_filtered_rad += 2 * math.pi

        # 转换回度
        final_yaw_deg = math.degrees(temp_filtered_rad)

        # 将平滑后的角度存入 sk 字典
        sk["body_yaw"] = final_yaw_deg

        # 添加这一行进行调试
        print(f"SkeletonSolver Output - Smoothed body_yaw: {final_yaw_deg:.3f}")

        sk["head"] = head
        return sk

    def reset_smoothing(self):
        """可选：重置平滑器状态"""
        self._previous_raw_angle_rad = None
        self._unwrapped_angle_rad = 0.0
        self._yaw_filter.prev = None  # 重置滤波器状态