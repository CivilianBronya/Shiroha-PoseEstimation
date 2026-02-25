import dlib, cv2, numpy as np

class HeadPose:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if not faces:
            return None

        shape = self.predictor(gray, faces[0])
        nose = (shape.part(30).x, shape.part(30).y)

        # 这里返回一个简化方向向量（后面算欧拉角）
        return nose
