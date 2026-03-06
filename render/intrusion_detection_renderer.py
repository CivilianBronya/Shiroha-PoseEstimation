import cv2
import numpy as np


class IntrusionDetectionRenderer:
    """
    入侵检测渲染器。
    此类接收多人姿态数据，为检测到的每个人绘制一个包围框，模拟入侵检测效果。
    """

    def __init__(self):
        self.color = (0, 0, 255)  # 红色
        self.thickness = 2
        self.expansion_factor = 1.2  # 扩展因子，使框更大一些

    def draw(self, frame, multi_body_data):
        """
        在帧上为检测到的每个人绘制一个包围框。

        Args:
            frame: 输入图像帧。
            multi_body_data: 一个包含多个人体关键点数据的列表。
                           格式应为: [person1_landmarks, person2_landmarks, ...]
                           其中每个 person_landmarks 是一个包含 [x, y, confidence] 的列表。

        Returns:
            True or False
        """
        if frame is None or not multi_body_data:
            return frame

        output_frame = frame.copy()

        for person_landmarks in multi_body_data:
            if not person_landmarks or len(person_landmarks) == 0:
                continue

            # 收集所有人身体关键点的坐标
            xs = []
            ys = []
            for point in person_landmarks:
                if point and len(point) >= 3:
                    x, y, conf = point
                    if conf > 0.1:
                        xs.append(x)
                        ys.append(y)

            if not xs or not ys:
                continue

            # 计算包围所有点的最小矩形
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))

            # 添加扩展
            width = x_max - x_min
            height = y_max - y_min

            # 计算扩展后的边界
            expansion = max(width, height) * (self.expansion_factor - 1) / 2
            x_min_expanded = int(x_min - expansion)
            x_max_expanded = int(x_max + expansion)
            y_min_expanded = int(y_min - expansion)
            y_max_expanded = int(y_max + expansion)

            # 确保坐标在图像范围内
            x_min_expanded = max(0, x_min_expanded)
            x_max_expanded = min(frame.shape[1], x_max_expanded)
            y_min_expanded = max(0, y_min_expanded)
            y_max_expanded = min(frame.shape[0], y_max_expanded)

            # 绘制更大的矩形框
            cv2.rectangle(output_frame, (x_min_expanded, y_min_expanded),
                          (x_max_expanded, y_max_expanded), self.color, self.thickness)

        return output_frame