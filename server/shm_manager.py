# -*- coding: utf-8 -*-
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

    def close(self):
        if self.shm:
            self.shm.close()
            # 只有在不再需要该内存块时调用 unlink (通常在 Flask 彻底关闭时)