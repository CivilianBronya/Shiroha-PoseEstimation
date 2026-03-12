# -*- coding: utf-8 -*-
"""
Shiroha WebSocket Server v2 - Concurrent Edition
"""
import asyncio
import websockets
import json
import logging
import sys
import signal
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from server.session import Session
from server.mjpeg_server import MjpegServer
from server.config import ServerConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ws_server")


class ShirohaServer:
    def __init__(self, config_path: str = "config.json"):
        self.config = ServerConfig.from_file(config_path)
        self.sessions = {}  # uuid -> Session
        self.mjpeg_server = MjpegServer(
            host=self.config.host,
            port=self.config.mjpeg_port
        )
        self._lock = asyncio.Lock()
        self._running = True

        # 🚀 线程池：用于处理 CPU 密集型的分析和渲染
        # 10人并发建议分配 16-20 个工作线程
        self.executor = ThreadPoolExecutor(max_workers=20)

        # 注册信号
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        logger.info(f"🛑 Shutdown signal received ({sig})")
        self._running = False

    # --- 线程池工作任务 ---
    def _decode_frame(self, frame_bytes):
        """在线程池中进行图像解码"""
        return cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

    def _process_and_render_task(self, session, frame):
        """在线程池中运行分析流水线并将渲染结果编码为 JPEG"""
        # 1. 执行算法逻辑
        result = session.process_frame(frame)

        # 2. 编码渲染帧
        render_jpegs = {}
        render_frames = result.get('render_frames', {})
        for mode, img_array in render_frames.items():
            if img_array is not None:
                success, buffer = cv2.imencode('.jpg', img_array, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if success:
                    render_jpegs[mode] = buffer.tobytes()

        return result.get('json_data', {}), render_jpegs

    # --- 协程方法 ---
    async def _cleanup_expired_sessions(self):
        """循环清理超时的不活跃会话"""
        while self._running:
            await asyncio.sleep(60)
            async with self._lock:
                now = time.time()
                expired = [
                    u for u, s in self.sessions.items()
                    if s.is_expired()
                ]
                for u in expired:
                    await self._remove_session(u)
                    logger.info(f"🗑️ Session expired: {u}")

    async def _remove_session(self, uuid: str):
        """安全移除会话及其资源"""
        if uuid in self.sessions:
            self.sessions[uuid].cleanup()
            del self.sessions[uuid]
            await self.mjpeg_server.unregister_buffer(uuid)

    async def handler(self, websocket, path=None):
        """WebSocket 核心处理器"""
        current_uuid = None
        loop = asyncio.get_running_loop()

        try:
            async for message in websocket:
                # 1. 处理文本消息 (控制/注册)
                if isinstance(message, str):
                    msg = json.loads(message)
                    action = msg.get('action')

                    if action == 'register':
                        u = msg.get('uuid')
                        if not u: continue

                        current_uuid = u.strip()
                        async with self._lock:
                            # 注册新 Session
                            session = Session(current_uuid, msg.get('config', {}), self.config)
                            session.websocket = websocket
                            self.sessions[current_uuid] = session
                            # 注册 MJPEG 缓冲区
                            await self.mjpeg_server.register_buffer(current_uuid)

                        logger.info(f"📋 Client Registered: {current_uuid}")
                        await websocket.send(json.dumps({'type': 'ack', 'uuid': current_uuid, 'status': 'registered'}))

                    elif action == 'ping':
                        await websocket.send(json.dumps({'type': 'pong'}))

                # 2. 处理二进制消息 (视频流)
                elif isinstance(message, bytes):
                    if not current_uuid or len(message) < 36:
                        continue

                    # 🚀 解析 UUID
                    try:
                        received_uuid = message[:36].decode('ascii').strip()
                        if received_uuid != current_uuid: continue
                        image_data = message[36:]
                    except:
                        continue

                    session = self.sessions.get(current_uuid)
                    if not session: continue
                    session.update_activity()

                    # 🚀 丢进线程池：解码 -> 分析 -> 绘图 -> JPEG 编码
                    try:
                        # a. 解码
                        img = await loop.run_in_executor(self.executor, self._decode_frame, image_data)
                        if img is None: continue

                        # b. 分析与渲染 (核心计算)
                        json_res, jpegs = await loop.run_in_executor(
                            self.executor, self._process_and_render_task, session, img
                        )

                        # c. 推送结果 (并发)
                        tasks = []
                        if json_res:
                            tasks.append(websocket.send(json.dumps({'type': 'update', 'data': json_res})))

                        for mode, jpeg_bytes in jpegs.items():
                            # MJPEG 推送现在是同步非阻塞的
                            self.mjpeg_server.push_frame(current_uuid, mode, jpeg_bytes)

                        if tasks:
                            await asyncio.gather(*tasks)

                    except Exception as e:
                        logger.error(f"❌ Processing Error ({current_uuid[:8]}): {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Connection closed: {current_uuid}")
        finally:
            if current_uuid:
                async with self._lock:
                    await self._remove_session(current_uuid)

    async def run(self):
        """启动主服务循环"""
        # 启动 MJPEG 服务器任务
        mjpeg_task = asyncio.create_task(self.mjpeg_server.run())
        # 启动清理任务
        cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())

        async with websockets.serve(
                self.handler,
                self.config.host,
                self.config.port,
                max_size=10 * 1024 * 1024
        ):
            logger.info(f"🚀 WS Server: ws://{self.config.host}:{self.config.port}")
            logger.info(f"📺 MJPEG Server: http://{self.config.host}:{self.config.mjpeg_port}")

            # 持续运行直到信号触发
            while self._running:
                await asyncio.sleep(1)

        cleanup_task.cancel()
        mjpeg_task.cancel()
        self.executor.shutdown(wait=True)
        logger.info("🛑 Server successfully stopped")


if __name__ == "__main__":
    server = ShirohaServer(config_path="config.json")
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass