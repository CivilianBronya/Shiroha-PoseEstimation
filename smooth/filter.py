import math

class LowPassFilter:
    def __init__(self, alpha=0.7, is_angle=False, angle_range=180):
        """
        初始化低通滤波器。

        Args:
            alpha (float): 滤波强度参数 (0 < alpha <= 1)。越大越平滑，响应越慢。
            is_angle (bool): 是否处理角度数据。如果是角度，会处理跨越边界的跳跃。
            angle_range (float): 角度的范围。180 表示 [-180, 180]，math.pi 表示 [-π, π]。
        """
        self.alpha = alpha
        self.is_angle = is_angle
        self.angle_range = angle_range
        self.prev = None

    def apply(self, value):
        """
        应用滤波器到输入值。

        Args:
            value: 输入值。

        Returns:
            float: 滤波后的值。
        """
        if self.prev is None:
            self.prev = value
            return value

        if self.is_angle:
            # 对角度值进行特殊处理
            filtered_value = self._apply_angle_filter(value)
        else:
            # 对普通数值进行滤波
            filtered_value = self.alpha * self.prev + (1 - self.alpha) * value

        self.prev = filtered_value
        return filtered_value

    def _apply_angle_filter(self, new_angle):
        """
        专门用于角度的滤波方法，处理 [-angle_range, angle_range] 范围内的角度跳跃。
        """
        # 计算当前滤波器输出与新输入的角度差
        delta_angle = new_angle - self.prev

        # 将角度差标准化到 [-range, range] 范围内
        full_circle = 2 * self.angle_range
        while delta_angle > self.angle_range:
            delta_angle -= full_circle
        while delta_angle <= -self.angle_range:
            delta_angle += full_circle

        # 对标准化后的角度差进行滤波
        filtered_delta = self.alpha * 0 + (1 - self.alpha) * delta_angle # 等价于 (1 - self.alpha) * delta_angle
        # 但这不是最终的滤波器更新方式。需要更新的是 prev 值。
        # 正确的滤波公式应用于 "prev + delta" 的形式：
        # new_filtered = alpha * old_filtered + (1-alpha) * new_input
        # 代入 new_input = prev + delta_angle
        # new_filtered = alpha * old_filtered + (1-alpha) * (prev + delta_angle)
        # 由于这里的 prev 是上一次的滤波输出，所以 old_filtered == prev
        # new_filtered = alpha * prev + (1-alpha) * (prev + delta_angle)
        # new_filtered = alpha * prev + (1-alpha) * prev + (1-alpha) * delta_angle
        # new_filtered = prev + (1-alpha) * delta_angle
        filtered_value = self.prev + (1 - self.alpha) * delta_angle

        # 将滤波后的角度重新映射回 [-angle_range, angle_range] 范围内
        while filtered_value > self.angle_range:
            filtered_value -= full_circle
        while filtered_value <= -self.angle_range:
            filtered_value += full_circle

        return filtered_value

    def reset(self):
        """重置滤波器状态。"""
        self.prev = None

    def set_alpha(self, new_alpha):
        """动态调整滤波强度参数。"""
        self.alpha = new_alpha
