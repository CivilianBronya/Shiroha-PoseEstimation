# -*- coding: utf-8 -*-
"""
Frame Buffer - Thread-safe latest frame storage
用于 MJPEG 流和 WebSocket 共享渲染结果
"""
import threading
import time
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FrameBuffer:
    """
    单会话帧缓冲区
    每个分析模式独立缓冲最新帧，自动过期清理
    """

    def __init__(self, max_age_sec: float = 10.0):
        self.max_age_sec = max_age_sec
        self._frames: Dict[str, tuple] = {}  # mode -> (jpeg_bytes, timestamp)
        self._lock = threading.Lock()

    def update(self, mode: str, jpeg_bytes: bytes):
        """更新某模式的最新帧"""
        with self._lock:
            self._frames[mode] = (jpeg_bytes, time.time())

    def get(self, mode: str) -> Optional[bytes]:
        """获取某模式的最新帧（超时返回 None）"""
        with self._lock:
            if mode not in self._frames:
                return None

            jpeg_bytes, timestamp = self._frames[mode]
            if time.time() - timestamp > self.max_age_sec:
                del self._frames[mode]
                return None

            return jpeg_bytes

    def get_latest(self) -> Dict[str, bytes]:
        """获取所有可用帧"""
        with self._lock:
            now = time.time()
            valid = {}
            expired = []

            for mode, (jpeg_bytes, timestamp) in self._frames.items():
                if now - timestamp <= self.max_age_sec:
                    valid[mode] = jpeg_bytes
                else:
                    expired.append(mode)

            for mode in expired:
                del self._frames[mode]

            return valid

    def clear(self):
        """清空缓冲区"""
        with self._lock:
            self._frames.clear()

    def list_modes(self) -> list:
        """列出有数据的模式"""
        with self._lock:
            return list(self._frames.keys())