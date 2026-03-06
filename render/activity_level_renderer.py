import cv2
import numpy as np


class ActivityLevelRenderer:
    def __init__(self, low_threshold=10.0, high_threshold=40.0):
        """
        初始化活动水平检测渲染器（单人版）。

        Args:
            low_threshold (float): 低活动水平的阈值。
            high_threshold (float): 高活动水平的阈值。
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.keypoint_history = {}

    def draw(self, frame, body_pts_list):
        """
        绘制活动水平检测的演示效果（单人版）。

        Args:
            frame: 当前视频帧 (BGR image)。
            body_pts_list: 单人姿态数据，列表格式，[[x, y], [x, y], ...]。

        Returns:
            True or False
        """
        debug_frame = frame.copy()

        if not body_pts_list or len(body_pts_list) == 0:
            # 如果没有检测到人体，清空历史并返回原图
            self.keypoint_history.clear()
            return debug_frame

        # 过滤掉无效的关键点
        valid_points = [(i, int(pt[0]), int(pt[1])) for i, pt in enumerate(body_pts_list) if pt[0] != 0 and pt[1] != 0]

        if not valid_points:
            # 如果所有点都无效，清空历史并返回原图
            self.keypoint_history.clear()
            return debug_frame

        _, x_coords, y_coords = zip(*valid_points)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # 更新每个关键点的历史位置
        for idx, x, y in valid_points:
            if idx not in self.keypoint_history:
                self.keypoint_history[idx] = []
            self.keypoint_history[idx].append((x, y))

            # 保持历史记录长度，例如只保留最近的几帧
            if len(self.keypoint_history[idx]) > 5:
                self.keypoint_history[idx].pop(0)

        level, avg_speed = self._assess_activity_level()

        # 根据活动水平选择颜色和标签
        if level == "LOW":
            box_color = (255, 0, 0)
            label = f"LOW ({avg_speed:.2f})"
        elif level == "MODERATE":
            box_color = (0, 255, 0)
            label = f"MOD ({avg_speed:.2f})"
        else:  # HIGH
            box_color = (0, 0, 255)
            label = f"HIGH ({avg_speed:.2f})"

        cv2.rectangle(debug_frame, (x_min, y_min), (x_max, y_max), box_color, thickness=2)

        # 绘制活动水平标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        # 获取文本框的尺寸
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        # 计算文本框的坐标
        text_origin_x = x_min
        text_origin_y = y_min - baseline - 1
        # 绘制文字背景和文字
        cv2.rectangle(debug_frame, (text_origin_x, text_origin_y - text_height),
                      (text_origin_x + text_width, text_origin_y + baseline), box_color, -1)
        cv2.putText(debug_frame, label, (text_origin_x, text_origin_y),
                    font, font_scale, color, thickness, cv2.LINE_AA)

        return debug_frame

    def _assess_activity_level(self):
        """评估整体活动水平"""
        total_speed = 0
        num_valid_points = 0

        for history in self.keypoint_history.values():
            if len(history) >= 2:
                # 计算最后一个位置与前一个位置的距离作为瞬时速度
                prev_point = np.array(history[-2])
                curr_point = np.array(history[-1])
                speed = np.linalg.norm(curr_point - prev_point)
                total_speed += speed
                num_valid_points += 1

        if num_valid_points == 0:
            return "LOW", 0.0

        average_speed = total_speed / num_valid_points

        if average_speed < self.low_threshold:
            return "LOW", average_speed
        elif average_speed < self.high_threshold:
            return "MODERATE", average_speed
        else:
            return "HIGH", average_speed