# server/shm_manager.py
import numpy as np
from multiprocessing import shared_memory

class ShmManager:
    def __init__(self, name="shiroha_frame", shape=(480, 640, 3)):
        self.name = name
        self.shape = shape
        # 🚀 修复点：强制转换为 Python 原生 int，防止 Windows _winapi 报错
        self.size = int(np.prod(shape) * np.dtype(np.uint8).itemsize)
        self.shm = None

    def create(self):
        """采集端调用：创建共享内存"""
        try:
            # 确保 size 是纯 int 类型
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
            print(f"✅ Created shared memory: {self.name} ({self.size} bytes)")
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.name)
            print(f"ℹ️ Attached to existing shared memory: {self.name}")
        return np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)

    def attach(self):
        """处理端调用：挂载已存在的共享内存"""
        try:
            self.shm = shared_memory.SharedMemory(name=self.name)
            return np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)
        except FileNotFoundError:
            print(f"❌ Error: Shared memory '{self.name}' not found. Start Flask first.")
            raise

    def close(self):
        if self.shm:
            self.shm.close()
            # 注意：在 Windows 上，unlink 可能会报错，通常 close 即可
            try:
                self.shm.unlink()
            except:
                pass