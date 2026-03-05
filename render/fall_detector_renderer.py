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

        risk_score = self.fall_detector.get_fall_risk_score()
        # TODO:将数据展示改为阈值，True与False将由前端做
        info_texts = [
            f"STATE: {state_name}",
            f"TILT: {tilt_val:.1f}°",
            f"VY: {vy_val:.2f} deg/s",
            f"SUPPORT: {support_val:.2f}",
            # f"FALL: {'TRUE' if is_falling else 'FALSE'}",
            # f"ALARM: {'TRUE' if is_falling else 'FALSE'}"
            f"SCORE: {risk_score:.2f}"
        ]

        x_offset = 10
        y_start_offset = 30

        for i, text in enumerate(info_texts):
            y_offset = y_start_offset + i * 30
            cv2.putText(dbg, text, (x_offset, y_offset), self.font, self.font_scale, self.text_color, self.thickness,
                        self.line_type)

        return dbg