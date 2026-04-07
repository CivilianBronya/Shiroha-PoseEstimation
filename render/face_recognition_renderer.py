# -*- coding: utf-8 -*-
import cv2


class FaceRecognitionRenderer:
    def __init__(self, face_solver_instance=None):
        self.thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2

    def draw(self, frame, processed_faces):
        if frame is None or not processed_faces:
            return frame

        for face_info in processed_faces.values():
            bbox = face_info.get('bbox')
            if not bbox:
                continue

            x, y, w, h = map(int, bbox)
            top_left = (x, y)
            bottom_right = (x + w, y + h)

            # 逻辑判断
            name = face_info.get('name', 'Unknown')
            distance = face_info.get('distance', 1.0)  # 如果数据里没有 distance，默认是陌生人

            if name == "Unknown":
                color = (0, 0, 255)  # 红色：陌生人 (未录入/识别失败)
                label_text = "STRANGER"
            elif distance < 0.4:
                color = (0, 255, 0)  # 绿色：已录入且高匹配 (通过)
                label_text = f"{name} "
            else:
                color = (0, 255, 255)  # 黄色：已录入但模糊 (需注意)
                label_text = f"{name}?"

            # 绘图
            cv2.rectangle(frame, top_left, bottom_right, color, self.thickness)

            # 绘制文字背景和文字
            (text_width, text_height), _ = cv2.getTextSize(label_text, self.font, self.font_scale, self.font_thickness)
            label_y = max(y, text_height + 10)
            cv2.rectangle(frame, (x, label_y - text_height - 10), (x + text_width + 10, label_y), color, -1)
            cv2.putText(frame, label_text, (x + 5, label_y - 5), self.font, self.font_scale, (0, 0, 0),
                        self.font_thickness)

        return frame