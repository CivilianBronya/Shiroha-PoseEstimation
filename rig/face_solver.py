import numpy as np


def calculate_iou(box1, box2):
    """计算两个边界框的交并比 (Intersection over Union)"""
    x1_1, y1_1, w1, h1 = box1
    x2_1, y2_1, w2, h2 = box2
    x1_2, y1_2 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x2_1 + w2, y2_1 + h2

    area1 = w1 * h1
    area2 = w2 * h2

    inter_x1 = max(x1_1, x2_1)
    inter_y1 = max(y1_1, y2_1)
    inter_x2 = min(x1_2, x2_2)
    inter_y2 = min(y1_2, y2_2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area


class TrackedFace:
    """代表一个被追踪的人脸对象"""

    def __init__(self, face_id, bbox, alpha):
        self.id = face_id
        self.bbox = bbox
        self.alpha = alpha
        self.center_history = [np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2])]
        self.disappeared_count = 0

    def update(self, new_bbox):
        self.disappeared_count = 0
        new_center = np.array([new_bbox[0] + new_bbox[2] / 2, new_bbox[1] + new_bbox[3] / 2])

        last_center = self.center_history[-1]
        filtered_center = self.alpha * new_center + (1 - self.alpha) * last_center

        self.center_history.append(filtered_center)
        if len(self.center_history) > 5:
            self.center_history.pop(0)

        self.bbox = [
            filtered_center[0] - new_bbox[2] / 2,
            filtered_center[1] - new_bbox[3] / 2,
            new_bbox[2],
            new_bbox[3]
        ]

    def mark_missing(self):
        self.disappeared_count += 1

    def get_filtered_bbox(self):
        if not self.center_history:
            return self.bbox
        cx, cy = self.center_history[-1]
        return [cx - self.bbox[2] / 2, cy - self.bbox[3] / 2, self.bbox[2], self.bbox[3]]


MODE_MULTI = "multi"
MODE_SINGLE = "single"

class FaceSolver:
    """
    人脸数据求解器。
    接收来自底层人脸检测组件的原始数据，并进行追踪和滤波，
    为渲染器提供平滑、稳定且ID一致的人脸位置信息。
    """

    def __init__(self, filter_alpha=0.7, max_disappeared=5, min_iou=0.3, min_area=2000, mode=MODE_MULTI):
        """
        Args:
            filter_alpha (float): 指数移动平均滤波器的参数。
            max_disappeared (int): 一个追踪对象在被删除前允许的最大连续消失帧数。
            min_iou (float): 用于匹配新检测结果和现有追踪对象的最小交并比阈值。
            min_area (int): 过滤掉面积小于该值的检测框，以减少误检。
            mode (str): 模式，'multi' 返回所有人脸，'single' 只返回一个置信度最高的人脸。
        """
        self.filter_alpha = filter_alpha
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou
        self.min_area = min_area
        self.mode = mode
        self.tracked_faces = {}
        self.next_id = 0

    def solve(self, raw_faces):
        """
        处理原始人脸数据。
        """
        # 过滤掉面积过小的检测框
        filtered_raw_faces = []
        for face_box in raw_faces:
            x, y, w, h = face_box
            area = w * h
            if area >= self.min_area:
                filtered_raw_faces.append(face_box)

        # 标记所有现有追踪对象为“失踪”
        for tracked_face in self.tracked_faces.values():
            tracked_face.mark_missing()

        # 匹配与更新
        if filtered_raw_faces:
            current_ids = list(self.tracked_faces.keys())
            current_predictions = [self.tracked_faces[tid].get_filtered_bbox() for tid in current_ids]

            if current_predictions:
                D = np.zeros((len(filtered_raw_faces), len(current_predictions)))
                for i, det in enumerate(filtered_raw_faces):
                    for j, pred in enumerate(current_predictions):
                        D[i, j] = calculate_iou(det, pred)

                matched_det_indices = set()
                matched_pred_indices = set()

                flat_indices = np.argsort(-D.ravel())
                row_indices, col_indices = np.unravel_index(flat_indices, D.shape)

                for det_idx, pred_idx in zip(row_indices, col_indices):
                    if det_idx in matched_det_indices or pred_idx in matched_pred_indices:
                        continue
                    if D[det_idx, pred_idx] < self.min_iou:
                        continue

                    face_id = current_ids[pred_idx]
                    self.tracked_faces[face_id].update(filtered_raw_faces[det_idx])

                    matched_det_indices.add(det_idx)
                    matched_pred_indices.add(pred_idx)

                for det_idx in range(len(filtered_raw_faces)):
                    if det_idx not in matched_det_indices:
                        new_tracked_face = TrackedFace(self.next_id, filtered_raw_faces[det_idx], self.filter_alpha)
                        self.tracked_faces[self.next_id] = new_tracked_face
                        self.next_id += 1
            else:
                for face_box in filtered_raw_faces:
                    new_tracked_face = TrackedFace(self.next_id, face_box, self.filter_alpha)
                    self.tracked_faces[self.next_id] = new_tracked_face
                    self.next_id += 1

        # 移除长时间未出现的追踪对象
        ids_to_delete = [fid for fid, tf in self.tracked_faces.items() if tf.disappeared_count > self.max_disappeared]
        for fid in ids_to_delete:
            del self.tracked_faces[fid]

        # 根据模式返回数据
        processed_faces = {}
        for face_id, tracked_face in self.tracked_faces.items():
            processed_faces[face_id] = {
                'bbox': tracked_face.get_filtered_bbox(),
                'center': tracked_face.get_filtered_bbox()[:2]
            }

        if self.mode == MODE_SINGLE and processed_faces:
            # 单人模式
            largest_face_id = max(processed_faces.keys(),
                                  key=lambda id: processed_faces[id]['bbox'][2] * processed_faces[id]['bbox'][3])
            return {largest_face_id: processed_faces[largest_face_id]}

        # 多人模式
        return processed_faces