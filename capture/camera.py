import cv2
import time
import threading
import queue

class Camera:
    def __init__(self, index=0, width=960, height=720, fps=30, buffer_size=1):
        """
        初始化摄像头。

        Args:
            index (int): 摄像头索引。
            width (int): 帧宽度。
            height (int): 帧高度。
            fps (int): 帧率。
            buffer_size (int): 内部队列缓冲区大小，1表示只保留最新帧。
        """
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        # 分辨率设置
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # 用于存储帧的队列
        self.frame_queue = queue.Queue(maxsize=buffer_size)

        # 控制线程的标志
        self.running = True

        # 启动读取帧的后台线程
        self.thread = threading.Thread(target=self._read_frames)
        self.thread.daemon = True
        self.thread.start()

        # 计算帧时间用的变量
        self.last_time = time.time()
        self.dt = 0

    def _read_frames(self):
        """在后台线程中持续读取帧并放入队列。"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            try:
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def read(self):
        """
        从队列中获取一帧。
        Returns:
            tuple: (frame, dt)。成功则返回(BGR图像帧, 时间间隔)，失败或队列为空时返回(None, 0)。
        """
        try:
            frame = self.frame_queue.get_nowait()

            now = time.time()
            dt = now - self.last_time
            self.last_time = now

            return frame, dt
        except queue.Empty:
            return None, 0

    def release(self):
        """停止线程并释放资源。"""
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()
    # TODO:流式传输
    # TODO:写多协议试图兼容其他类的如摄像头，手表，手机，还有小智（以演示为主）