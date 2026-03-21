# -*- coding: utf-8 -*-
import cv2
import numpy as np
from multiprocessing import shared_memory
import platform


class ShmManager:
    def __init__(self, name="shiroha_frame", shape=(480, 640, 3)):
        self.name = name
        self.shape = shape
        # 🚀 强制转换为原生 int，修复 Windows CreateFileMapping 报错
        self.size = int(np.prod(shape) * np.dtype(np.uint8).itemsize)
        self.shm = None
        # 🚀 必须在这里显式声明，防止 AttributeError
        self._buffer = None

    def create(self):
        """采集端(Flask)调用：创建共享内存"""
        try:
            # 如果是 Linux，尝试清理残留
            if platform.system() != "Windows":
                try:
                    old_shm = shared_memory.SharedMemory(name=self.name)
                    old_shm.close()
                    old_shm.unlink()
                except:
                    pass

            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
            print(f"✅ 共享内存创建成功: {self.name} ({self.size} bytes)")
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.name)
            print(f"ℹ️ 共享内存已存在，直接挂载: {self.name}")

        return np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)

    def attach(self):
        """处理端(rtc_main)调用：挂载内存"""
        try:
            self.shm = shared_memory.SharedMemory(name=self.name)
            return np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)
        except FileNotFoundError:
            # 向上抛出异常，由逻辑层决定是否重试
            raise FileNotFoundError(f"未找到共享内存 '{self.name}'。请先启动 Flask 程序。")

    def write(self, frame):
        """将图像帧写入共享内存"""
        # 🚀 健壮性检查：如果执行 write 时还没有 buffer，尝试自动关联
        if self._buffer is None:
            if self.shm is not None:
                # 如果 shm 对象存在但没有 ndarray，重新包装
                self._buffer = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)
            else:
                # 这种情况通常说明没调用 create()，直接忽略这次写入，防止崩溃
                return

        if frame is not None:
            try:
                # 确保尺寸一致
                if frame.shape[0] != self.shape[0] or frame.shape[1] != self.shape[1]:
                    frame = cv2.resize(frame, (self.shape[1], self.shape[0]))

                # 真正的写入操作
                self._buffer[:] = frame[:]
            except Exception as e:
                print(f"❌ SHM 写入运行时出错: {e}")

    def close(self):
        if self.shm:
            self.shm.close()
            # 只有在不再需要该内存块时调用 unlink (通常在 Flask 彻底关闭时)
