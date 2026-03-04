import cv2
from analysis.fall_detector import FallDetector

class FallDetectorRenderer:
    def __init__(self, fall_detector=None):
        """
        初始化摔倒检测渲染器。
        :param fall_detector: 可选的 FallDetector 实例。如果未提供，则内部创建一个。
        """
        self.fall_detector = fall_detector or FallDetector(ground_threshold_sec=4.5)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 1
        self.text_color = (255, 255, 255)  # 白色文字
        self.bg_color = (0, 0, 0)          # 黑色背景
        self.line_type = cv2.LINE_AA

    def draw(self, frame, skeleton):
        """
        在帧上绘制摔倒检测的状态信息。

        Args:
            frame: 输入图像帧。
            skeleton: 来自 SkeletonSolver 的骨架数据。

        Returns:
            绘制了状态信息的图像帧。
        """
        if frame is None:
            return frame

        dbg = frame.copy()

        # 摔倒检测器处理骨架数据，并返回是否摔倒的布尔值
        is_falling = self.fall_detector.update(skeleton)

        # 获取当前状态名称
        state_name = self.fall_detector.get_state_name()

        # 获取最新特征
        current_features = self.fall_detector.get_last_features()
        tilt_val = current_features.get('tilt', 0)
        vy_val = current_features.get('vy', 0)
        support_val = current_features.get('support', 1)

        info_texts = [
            f"STATE: {state_name}",
            f"TILT: {tilt_val:.1f}°",
            f"VY: {vy_val:.2f} deg/s",
            f"SUPPORT: {support_val:.2f}",
            f"FALL: {'TRUE' if is_falling else 'FALSE'}",
            f"ALARM: {'TRUE' if is_falling else 'FALSE'}"
        ]

        x_offset = 10
        y_start_offset = 30
        (text_w, text_h), _ = cv2.getTextSize(info_texts[0], self.font, self.font_scale, self.thickness)

        num_lines = len(info_texts)
        rect_top_left = (x_offset - 5, y_start_offset - (text_h + 5))
        rect_bottom_right = (x_offset + 220, y_start_offset + (text_h + 5) * num_lines)
        cv2.rectangle(dbg, rect_top_left, rect_bottom_right, self.bg_color, -1, self.line_type)

        for i, text in enumerate(info_texts):
            y_offset = y_start_offset + i * (text_h + 5)
            color_to_use = self.text_color
            if text.startswith("FALL:") or text.startswith("ALARM:"):
                color_to_use = (0, 0, 255) if is_falling else (0, 255, 0)
            cv2.putText(dbg, text, (x_offset, y_offset), self.font, self.font_scale, color_to_use, self.thickness, self.line_type)

        return dbg