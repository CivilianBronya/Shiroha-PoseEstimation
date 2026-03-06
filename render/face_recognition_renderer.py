import cv2
import numpy as np

class FaceRecognitionRenderer:
    """
    人脸识别渲染器。
    此类接收一个 FaceSolver 实例，并从中获取处理后的人脸数据，
    在帧上绘制人脸框。
    """

    def __init__(self, face_solver_instance):
        self.face_solver = face_solver_instance
        self.color = (0, 255, 0)
        self.thickness = 2

    def draw(self, frame, faces_data):
        """
        在帧上绘制人脸框。

        Args:
            frame: 输入图像帧。
            faces_data: 人脸检测组件返回的原始人脸数据。
                       例如: [[x, y, w, h], ...]

        Returns:
            True or False
        """
        if frame is None:
            return frame

        output_frame = frame.copy()

        # 将原始数据传递给人脸求解器进行处理
        processed_faces = self.face_solver.solve(faces_data)

        # 从求解器获取平滑后的人脸位置并绘制
        for face_info in processed_faces.values():
            bbox = face_info['bbox']
            top_left = (int(bbox[0]), int(bbox[1]))
            bottom_right = (int(bbox[0] + bbox[2]), int(bbox[3] + bbox[1]))

            cv2.rectangle(output_frame, top_left, bottom_right, self.color, self.thickness)

        return output_frame