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
        self.font_scale = 0.5
        self.thickness = 1
        self.text_color = (255, 255, 255)
        self.line_type = cv2.LINE_AA
        self.bg_color = (0, 0, 0)
        self.bg_alpha = 0.4

    def draw(self, frame, skeleton, original_keypoints_2d=None):
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
        self.fall_detector.update(skeleton, original_keypoints_2d)
        # print("Debug: original_keypoints_2d is None:", original_keypoints_2d is None)
        # if original_keypoints_2d is not None:
            # print("Debug: First keypoint (x,y,conf):", original_keypoints_2d[0])

        # 获取当前状态名称
        state_name = self.fall_detector.get_state_name()

        # 获取最新特征
        current_features = self.fall_detector.get_last_features()
        tilt_val = current_features.get('tilt', 0)
        vy_val = current_features.get('vy', 0)
        support_val = current_features.get('support', 1)

        risk_score = self.fall_detector.get_fall_risk_score()
        score_display = f"SCORE: {risk_score}"
        info_texts = [
            f"STATE: {state_name}",
            f"TILT: {tilt_val:.1f}°",
            f"VY: {vy_val:.2f} px/s",
            f"SUPPORT: {support_val:.2f}",
            score_display
        ]

        x_offset = 10
        y_start_offset = 30

        for i, text in enumerate(info_texts):
            y_offset = y_start_offset + i * 30

            text_size = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)[0]
            text_width, text_height = text_size[0], text_size[1]

            bg_x1 = x_offset - 5
            bg_y1 = y_offset - text_height // 2 - 5
            bg_x2 = x_offset + text_width + 5
            bg_y2 = y_offset + text_height // 2 + 5

            overlay = dbg.copy()

            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), self.bg_color, -1)  # -1 填充

            cv2.addWeighted(overlay, self.bg_alpha, dbg, 1 - self.bg_alpha, 0, dbg)

            cv2.putText(dbg, text, (x_offset, y_offset), self.font, self.font_scale, self.text_color, self.thickness,
                        self.line_type)

        return dbg