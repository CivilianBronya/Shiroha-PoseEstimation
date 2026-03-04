import cv2
import time

class Camera:
    def __init__(self, index=0, width=960, height=720, fps=30):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        # 分辨率（太高会卡，太低骨架抖）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.last = time.time()
        self.dt = 0

    # 读取帧
    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # 计算帧时间（后面滤波/速度会用）
        now = time.time()
        self.dt = now - self.last
        self.last = now

        return frame

    # 显示
    def show(self, frame, name="MotionCapture"):
        if frame is None:
            return False

        # 左上角显示FPS
        if self.dt > 0:
            fps = int(1/self.dt)
            cv2.putText(frame, f"FPS:{fps}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow(name, frame)

        # ESC退出
        key = cv2.waitKey(1)
        if key == 27:
            self.release()
            return False
        return True

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
    # TODO:写多协议试图兼容其他类的如摄像头，手表，手机，还有小智（以演示为主）