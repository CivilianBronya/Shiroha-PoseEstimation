# server/frame_buffer.py
import asyncio
import threading
import time

class FrameBuffer:
    def __init__(self, max_age_sec: float = 5.0):
        self.max_age_sec = max_age_sec
        self._frames = {}
        self._lock = threading.Lock()
        # 🚀 核心：为每个模式创建一个异步事件，用于通知新帧到达
        self.events = {}

    def update(self, mode: str, jpeg_bytes: bytes):
        with self._lock:
            self._frames[mode] = (jpeg_bytes, time.time())
            # 如果该模式有等待的事件，触发它
            if mode in self.events:
                # 注意：Event.set() 需要在主事件循环中执行，后面在 push_frame 处处理
                pass

    def get(self, mode: str):
        with self._lock:
            if mode not in self._frames: return None
            data, ts = self._frames[mode]
            if time.time() - ts > self.max_age_sec: return None
            return data