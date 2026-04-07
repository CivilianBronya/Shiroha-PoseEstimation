import dlib
import numpy as np
import cv2


class FaceRecognitionAnalyzer:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

        try:
            self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
            self.face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
        except RuntimeError:
            print("错误：无法加载 Dlib 模型文件，请检查路径是否正确。")
            return

        # 预存已知人脸
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """加载已知人脸图片并计算特征"""
        # 比如 known_faces/a.jpg, known_faces/b.jpg
        import os
        if not os.path.exists("known_faces"): return

        for file_name in os.listdir("known_faces"):
            path = os.path.join("known_faces", file_name)
            img = dlib.load_rgb_image(path)
            # 获取人脸位置
            faces = self.detector(img, 1)
            if len(faces) > 0:
                # 获取关键点
                shape = self.predictor(img, faces[0])
                # 计算特征向量 (128D)
                face_encoding = self.face_rec_model.compute_face_descriptor(img, shape)
                self.known_face_encodings.append(np.array(face_encoding))
                self.known_face_names.append(file_name.split('.')[0])  # 文件名作为名字
                print(f"加载人脸: {file_name}")

    def solve(self, frame):
        """
        分析帧，返回处理后的数据
        返回格式需适配 Renderer: {id: {'bbox': [x,y,w,h], 'name': 'Name'}}
        """
        results = {}

        # 转灰度图加速检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸 (rectangles)
        rects = self.detector(gray, 0)

        for i, rect in enumerate(rects):
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            bbox = [x, y, w, h]

            # 获取关键点 (用于对齐)
            shape = self.predictor(gray, rect)

            # 计算特征向量
            face_descriptor = self.face_rec_model.compute_face_descriptor(frame, shape)
            current_encoding = np.array(face_descriptor)

            # 比对 (计算欧氏距离)
            name = "Unknown"
            min_dist = 999.0

            for known_enc, known_name in zip(self.known_face_encodings, self.known_face_names):
                dist = np.linalg.norm(current_encoding - known_enc)
                if dist < min_dist and dist < 0.6:
                    min_dist = dist
                    name = known_name

            # 使用 i 作为临时 ID
            results[i] = {
                'bbox': bbox,
                'name': name,
                'confidence': 1.0 - min_dist,
                'distance': min_dist
            }

        return results