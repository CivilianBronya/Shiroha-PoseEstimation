import cv2
import numpy as np


class VigorousActivityRenderer:
    def __init__(self, activity_threshold=50.0):
        """
        初始化剧烈活动检测渲染器（单人版）。

        Args:
            activity_threshold (float): 判断为剧烈活动的速度阈值。
        """
        self.activity_threshold = activity_threshold
        self.keypoint_history = {}
        self.is_currently_active = False

    def draw(self, frame, body_pts_list):
        """
        绘制剧烈活动检测的演示效果（单人版）。

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
            self.is_currently_active = False
            return debug_frame

        # 过滤掉无效的关键点
        valid_points = [(i, int(pt[0]), int(pt[1])) for i, pt in enumerate(body_pts_list) if pt[0] != 0 and pt[1] != 0]

        if not valid_points:
            self.keypoint_history.clear()
            self.is_currently_active = False
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

        # 检查剧烈活动
        is_vigorous = self._is_person_vigorous()

        box_color = (0, 0, 255) if is_vigorous else (0, 255, 0)  # 剧烈活动为红，否则为绿
        cv2.rectangle(debug_frame, (x_min, y_min), (x_max, y_max), box_color, thickness=2)

        if is_vigorous:
            label = "VIGOROUS"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (255, 255, 255)
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            text_origin_x = x_min
            text_origin_y = y_min - baseline - 1
            cv2.rectangle(debug_frame, (text_origin_x, text_origin_y - text_height),
                          (text_origin_x + text_width, text_origin_y + baseline), box_color, -1)
            cv2.putText(debug_frame, label, (text_origin_x, text_origin_y),
                        font, font_scale, color, thickness, cv2.LINE_AA)

        self.is_currently_active = is_vigorous

        return debug_frame

    def _is_person_vigorous(self):
        """判断一个人是否处于剧烈活动中"""
        total_speed = 0
        num_valid_points = 0

        for keypoint_id, history in self.keypoint_history.items():
            if len(history) >= 2:
                # 计算最后一个位置与前一个位置的距离作为瞬时速度
                prev_point = np.array(history[-2])
                curr_point = np.array(history[-1])
                speed = np.linalg.norm(curr_point - prev_point)
                total_speed += speed
                num_valid_points += 1

        if num_valid_points == 0:
            return False

        average_speed = total_speed / num_valid_points

        return average_speed > self.activity_threshold