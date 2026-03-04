import cv2
import numpy as np

from .single_stickman_renderer import SingleStickmanRenderer


class MultiStickmanRenderer:
    def __init__(self):
        self.single_renderer = SingleStickmanRenderer()
        self.bone_color = (0, 255, 0)
        self.joint_color = (0, 200, 255)
        self.box_color = (255, 0, 0)
        self.id_bg_color = (0, 0, 0, 128)
        self.id_text_color = (255, 255, 255)
        self.aim_line_color = (255, 0, 0)

    def draw(self, frame, multi_body_data):
        """
        绘制多个人的骨架、外接框、ID标签和瞄准线。

        Args:
            frame: 输入图像帧
            multi_body_data: 多人的身体关键点列表 [[(x,y),...], [...]]

        Returns:
            绘制后的图像帧。
        """
        if frame is None or not multi_body_data:
            return frame

        overlay = frame.copy()
        alpha = 0.8

        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2

        for i, body_pts_list in enumerate(multi_body_data):
            if body_pts_list is None or len(body_pts_list) == 0:
                continue

            # 获取所有非空的关键点坐标
            points = [(int(pt[0]), int(pt[1])) for pt in body_pts_list if pt is not None]

            if len(points) < 2:
                continue

            points_np = np.array(points, dtype=np.int32)

            # 计算最小外接矩形
            rect = cv2.minAreaRect(points_np)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)

            # 绘制外接框
            cv2.drawContours(frame, [box], 0, self.box_color, 2)

            x_min = min([p[0] for p in box])
            y_min = min([p[1] for p in box])
            label_pos = (x_min - 15, y_min - 10)
            id_text = f"User{i + 1}"
            (text_width, text_height), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            bg_top_left = (label_pos[0], label_pos[1] - text_height - 5)
            bg_bottom_right = (label_pos[0] + text_width + 5, label_pos[1] + baseline)
            cv2.rectangle(overlay, bg_top_left, bg_bottom_right, (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # 在原图上绘制文字
            cv2.putText(frame, id_text, (bg_top_left[0] + 2, bg_top_left[1] + text_height + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.id_text_color, 1)

            text_center_x = label_pos[0] + text_width // 2
            text_center_y = label_pos[1] + text_height // 2
            cv2.line(frame, (text_center_x, text_center_y), box[0], self.aim_line_color, 1, cv2.LINE_AA)

            # 绘制骨骼
            for a, b in SingleStickmanRenderer.BONES:
                if a < len(body_pts_list) and b < len(body_pts_list):
                    pt_a = body_pts_list[a]
                    pt_b = body_pts_list[b]
                    if pt_a is not None and pt_b is not None:
                        pa = (int(pt_a[0]), int(pt_a[1]))
                        pb = (int(pt_b[0]), int(pt_b[1]))
                        cv2.line(frame, pa, pb, self.bone_color, 1)

            # 绘制关节点
            for p in body_pts_list:
                if p is not None:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 2, self.joint_color, -1)

        return frame