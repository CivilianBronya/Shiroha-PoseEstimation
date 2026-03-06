import cv2
import numpy as np


class LoiteringDetectionRenderer:
    def __init__(self, alert_duration=15, cycle_length=90, alert_threshold=10):
        self._frame_counter = 0
        self._alert_duration = alert_duration
        self._cycle_length = cycle_length
        self._alert_threshold = alert_threshold
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 1
        self.text_color = (255, 255, 255)
        self.line_type = cv2.LINE_AA
        self.box_color = (0, 0, 255)

    def set_alert_threshold(self, threshold):
        """
        设置警报触发的阈值。

        Args:
            threshold: 触发警报所需的帧数。
        """
        self._alert_threshold = threshold

    def draw(self, frame, multi_body_data):
        """
        绘制徘徊检测的演示效果。

        Args:
            frame: 当前视频帧 (BGR image)。
            multi_body_data: 来自多人姿态检测的数据，列表格式，[[pt1, pt2, ...], ...]。

        Returns:
            True or False
        """
        self._frame_counter = (self._frame_counter + 1) % self._cycle_length
        debug_frame = frame.copy()

        if self._frame_counter < self._alert_threshold:
            self._draw_alert_text(debug_frame)

        if multi_body_data and len(multi_body_data) > 0:
            body_pts_list = multi_body_data[0]

            valid_points = [(int(pt[0]), int(pt[1])) for pt in body_pts_list if pt[0] != 0 and pt[1] != 0]

            if len(valid_points) > 0:
                x_coords, y_coords = zip(*valid_points)
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                cv2.rectangle(debug_frame, (x_min, y_min), (x_max, y_max), self.box_color, thickness=2)

        return debug_frame

    def _draw_alert_text(self, frame):
        """
        在帧上绘制警报文本。
        """
        alert_text = "ALERT: LOITERING DETECTED!"
        text_x = 10
        text_y = 30
        cv2.putText(frame, alert_text, (text_x, text_y), self.font, self.font_scale, self.text_color, self.thickness,
                    self.line_type)