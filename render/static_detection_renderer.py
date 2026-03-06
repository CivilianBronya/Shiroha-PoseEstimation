import cv2
import numpy as np


class StaticDetectionRenderer:
    def __init__(self, history_length=30, movement_threshold=10.0):
        """
        初始化静态检测渲染器。

        Args:
            history_length (int): 保存历史位置的帧数。
            movement_threshold (float): 判断为静止的最大位移阈值。
        """
        self.history_length = history_length
        self.movement_threshold = movement_threshold
        self.person_history = {}
        self.next_track_id = 0

    def draw(self, frame, multi_body_data):
        """
        绘制静止检测的演示效果。

        Args:
            frame: 当前视频帧 (BGR image)。
            multi_body_data: 来自多人姿态检测的数据，列表格式，[[pt1, pt2, ...], ...]。

        Returns:
            True or False
        """
        debug_frame = frame.copy().copy()
        current_centers = {}
        used_ids = set()

        for body_pts_list in multi_body_data:
            if len(body_pts_list) > 0:
                valid_points = [(int(pt[0]), int(pt[1])) for pt in body_pts_list if pt[0] != 0 and pt[1] != 0]

                if len(valid_points) > 0:
                    x_coords, y_coords = zip(*valid_points)
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    # 计算人体框的中心点
                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2
                    current_center = (center_x, center_y)

                    # 简单的跟踪逻辑：寻找最近的历史ID
                    best_id = self._find_closest_id(current_center, used_ids)

                    if best_id is None:
                        # 如果没有找到匹配的ID，则分配一个新的ID
                        best_id = self.next_track_id
                        self.next_track_id += 1

                    used_ids.add(best_id)

                    # 更新历史记录
                    if best_id not in self.person_history:
                        self.person_history[best_id] = []
                    self.person_history[best_id].append(current_center)

                    # 保持历史记录长度
                    if len(self.person_history[best_id]) > self.history_length:
                        self.person_history[best_id].pop(0)

                    # 绘制人体框
                    is_static = self._is_person_static(self.person_history[best_id])
                    box_color = (255, 0, 0) if is_static else (0, 255, 0)  # 静止为蓝，移动为绿
                    cv2.rectangle(debug_frame, (x_min, y_min), (x_max, y_max), box_color, thickness=2)

                    # 如果是静止的，绘制文字
                    if is_static:
                        label = "STATIC"
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

        # 清理不再出现的跟踪ID
        ids_to_remove = [tid for tid in self.person_history if tid not in used_ids]
        for tid in ids_to_remove:
            del self.person_history[tid]

        return debug_frame

    def _find_closest_id(self, center, used_ids):
        """寻找与当前中心点最近的历史ID"""
        min_dist = float('inf')
        best_id = None
        for tid, history in self.person_history.items():
            if tid in used_ids:
                continue
            if history:
                last_center = history[-1]
                dist = np.linalg.norm(np.array(center) - np.array(last_center))
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid
        return best_id if min_dist < self.movement_threshold * 2 else None

    def _is_person_static(self, history):
        """判断一个人是否静止"""
        if len(history) < self.history_length:
            return False

        start_point = np.array(history[0])
        end_point = np.array(history[-1])
        total_movement = np.linalg.norm(end_point - start_point)

        return total_movement < self.movement_threshold