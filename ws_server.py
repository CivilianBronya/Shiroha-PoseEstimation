# -*- coding: utf-8 -*-
"""
Shiroha WebSocket Server v2 - MJPEG Edition
功能：
  - WebSocket: 接收视频帧 + 推送 JSON 状态（控制通道）
  - HTTP MJPEG: 浏览器 <img> 直接拉取渲染结果（视频通道）
  - UUID 多租户隔离，Ubuntu 24 headless 兼容
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
from server.session import Session
from server.mjpeg_server import MjpegServer
from server.config import ServerConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('shiroha_ws.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("ws_server")


class ShirohaServer:
    def __init__(self, config_path: str = "config.json"):
        self.config = ServerConfig.from_file(config_path)
        self.sessions = {}  # uuid -> Session
        self.mjpeg_server = MjpegServer(
            host=self.config.host,
            port=self.config.get('mjpeg_port', 8766)
        )
        self._lock = asyncio.Lock()
        self._running = True

        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        logger.info(f"🛑 Received signal {sig}, shutting down...")
        self._running = False

    async def _cleanup_expired_sessions(self):
        """后台任务：清理超时会话"""
        while self._running:
            await asyncio.sleep(60)
            async with self._lock:
                expired = [
                    uuid for uuid, sess in self.sessions.items()
                    if sess.is_expired()
                ]
                for uuid in expired:
                    await self._remove_session(uuid)
                    logger.info(f"🗑️ Cleaned expired session: {uuid}")

    async def _remove_session(self, uuid: str):
        """安全移除会话"""
        if uuid in self.sessions:
            self.sessions[uuid].cleanup()
            del self.sessions[uuid]
            # 清理 MJPEG 缓冲区
            await self.mjpeg_server.unregister_buffer(uuid)

    async def _send_to_client(self, websocket, data: dict):
        """安全发送消息"""
        try:
            await websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            raise
        except Exception as e:
            logger.warning(f"Send error: {e}")

    async def handler(self, websocket, path=None):
        """WebSocket 连接处理器"""
        current_uuid = None
        session = None

        try:
            logger.info(f"🔗 New connection from {websocket.remote_address}")

            async for message in websocket:
                # === 文本消息：控制命令 ===
                if isinstance(message, str):
                    try:
                        msg = json.loads(message)
                        action = msg.get('action')

                        if action == 'register':
                            # 🔑 从注册消息获取 UUID
                            current_uuid = msg.get('uuid')
                            if not current_uuid:
                                await self._send_to_client(websocket, {
                                    'type': 'error',
                                    'message': 'UUID required'
                                })
                                continue

                            config = msg.get('config', {})

                            async with self._lock:
                                if current_uuid in self.sessions:
                                    logger.warning(f"UUID {current_uuid} already registered")
                                    await self._send_to_client(websocket, {
                                        'type': 'error',
                                        'message': 'UUID already registered'
                                    })
                                    continue

                                # 🔑 创建 Session
                                session = Session(current_uuid, config, self.config)
                                session.websocket = websocket
                                self.sessions[current_uuid] = session

                            # 🔑 注册 MJPEG 缓冲区
                            await self.mjpeg_server.register_buffer(current_uuid)

                            logger.info(f"📋 Registered: {current_uuid}")
                            await self._send_to_client(websocket, {
                                'type': 'ack',
                                'uuid': current_uuid,
                                'status': 'registered',
                                'modes': list(session.subscribed_modes)
                            })

                        elif action == 'ping':
                            await self._send_to_client(websocket, {'type': 'pong'})

                        elif action == 'unsubscribe':
                            mode = msg.get('mode')
                            if session and mode in session.subscribed_modes:
                                session.subscribed_modes.remove(mode)
                                logger.info(f"🔕 {current_uuid} unsubscribed: {mode}")

                        elif action == 'resubscribe':
                            mode = msg.get('mode')
                            if session and mode in self.config.modes:
                                session.subscribed_modes.add(mode)
                                logger.info(f"🔔 {current_uuid} resubscribed: {mode}")

                        else:
                            logger.warning(f"❓ Unknown action: {action}")

                    except json.JSONDecodeError as e:
                        logger.error(f"❌ JSON decode error: {e}")

                # === 二进制消息：视频帧 ===
                elif isinstance(message, bytes):
                    if not current_uuid or current_uuid not in self.sessions:
                        logger.warning("❌ Frame received before registration")
                        continue

                    if len(message) < 36:
                        logger.warning("❌ Binary message too short")
                        continue

                    # 🔑 解析：[36 字节 UUID][JPEG 数据]
                    uuid_bytes = message[:36]
                    frame_bytes = message[36:]

                    try:
                        received_uuid = uuid_bytes.decode('ascii', errors='ignore').strip()
                    except Exception as e:
                        logger.error(f"❌ UUID decode error: {e}")
                        continue

                    if received_uuid != current_uuid:
                        logger.warning(f"⚠️ UUID mismatch: {received_uuid} != {current_uuid}")
                        continue

                    session = self.sessions[current_uuid]
                    session.update_activity()

                    # 🔑 解码帧
                    try:
                        frame = cv2.imdecode(
                            np.frombuffer(frame_bytes, np.uint8),
                            cv2.IMREAD_COLOR
                        )
                        if frame is None:
                            raise ValueError("Decode failed")
                    except Exception as e:
                        logger.error(f"❌ Frame decode error: {e}")
                        continue

                    # 🔑 执行分析流水线
                    result = session.process_frame(frame)

                    # 🔑 推送渲染帧到 MJPEG 缓冲区（使用 numpy 数组！）
                    render_frames = result.get('render_frames', {})  # ← 读取 numpy 数组
                    for mode, render_frame in render_frames.items():
                        try:
                            # ✅ 验证 numpy 数组
                            if render_frame is None or render_frame.size == 0:
                                continue
                            if not isinstance(render_frame, np.ndarray):
                                logger.warning(f"⚠️ Invalid frame type for {mode}: {type(render_frame)}")
                                continue

                            # ✅ 编码为 JPEG
                            _, jpeg = cv2.imencode('.jpg', render_frame,
                                                   [cv2.IMWRITE_JPEG_QUALITY,
                                                    self.config.get('jpeg_quality', 75)])

                            # ✅ 异步推送到 MJPEG 缓冲区
                            asyncio.create_task(
                                self.mjpeg_server.push_frame(
                                    current_uuid,
                                    mode,
                                    jpeg.tobytes()
                                )
                            )
                        except Exception as e:
                            logger.warning(f"⚠️ MJPEG push error [{mode}]: {e}", exc_info=True)

                    # 🔑 只通过 WebSocket 推送 JSON 状态（不推图像）
                    if result.get('status') and session.output_json:
                        await self._send_to_client(websocket, {
                            'uuid': current_uuid,
                            'type': 'status',
                            'data': result['status']
                        })

                else:
                    logger.warning(f"❓ Unknown message type: {type(message)}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"🔌 Connection closed: {current_uuid} (code={e.code})")
        except Exception as e:
            logger.error(f"💥 Handler error:", exc_info=True)
        finally:
            if current_uuid:
                async with self._lock:
                    await self._remove_session(current_uuid)

    async def run(self):
        """启动服务"""
        logger.info(f"🚀 Starting Shiroha Server at ws://{self.config.host}:{self.config.port}")
        logger.info(f"📺 MJPEG Server at http://{self.config.host}:{self.mjpeg_server.port}")
        logger.info(f"   Modes: {self.config.modes}")

        # 启动 MJPEG 服务器（后台任务）
        mjpeg_task = asyncio.create_task(self.mjpeg_server.run())
        cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())

        try:
            async with websockets.serve(
                    self.handler,
                    self.config.host,
                    self.config.port,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=2 ** 24  # 16MB max frame
            ) as server:
                logger.info("✅ WebSocket + MJPEG servers RUNNING!")
                while self._running:
                    await asyncio.sleep(1)
        finally:
            self._running = False
            mjpeg_task.cancel()
            cleanup_task.cancel()
            logger.info("🛑 Server stopped")


async def main():
    server = ShirohaServer("./config.json")
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
