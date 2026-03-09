# -*- coding: utf-8 -*-
"""
Shiroha WebSocket 服务端 v3 - 生产就绪版
功能：
  - 多 UUID 会话隔离，复用原 main.py 分析逻辑
  - 输出：JSON 状态 + 10 种渲染图像（base64）
  - Ubuntu 24 headless 兼容（无 cv2.imshow）
"""
import asyncio
import websockets
import json
import logging
import sys
import signal
import time
from server.session import Session
from server.config import ServerConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('shiroha_server.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("ws_server")


class ShirohaServer:
    def __init__(self, config_path: str = "config.json"):
        self.config = ServerConfig.from_file(config_path)
        self.sessions = {}  # uuid -> Session
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
        session: Session = None

        try:
            logger.info(f"🔗 New connection from {websocket.remote_address}")

            async for message in websocket:
                # === 文本消息：控制命令 ===
                if isinstance(message, str):
                    try:
                        msg = json.loads(message)
                        action = msg.get('action')

                        if action == 'register':
                            uuid = msg.get('uuid')
                            if not uuid:
                                await self._send_to_client(websocket, {
                                    'type': 'error',
                                    'message': 'UUID required'
                                })
                                continue

                            config = msg.get('config', {})
                            async with self._lock:
                                if uuid in self.sessions:
                                    logger.warning(f"UUID {uuid} already registered")
                                    await self._send_to_client(websocket, {
                                        'type': 'error',
                                        'message': 'UUID already registered'
                                    })
                                    continue
                                session = Session(uuid, config, self.config)
                                session.websocket = websocket
                                self.sessions[uuid] = session

                            logger.info(f"📋 Registered: {uuid}")
                            await self._send_to_client(websocket, {
                                'type': 'ack',
                                'uuid': uuid,
                                'status': 'registered',
                                'modes': list(session.subscribed_modes)
                            })

                        elif action == 'ping':
                            await self._send_to_client(websocket, {'type': 'pong'})

                        elif action == 'unsubscribe':
                            mode = msg.get('mode')
                            if session and mode in session.subscribed_modes:
                                session.subscribed_modes.remove(mode)
                                logger.info(f"🔕 {session.uuid} unsubscribed: {mode}")

                        elif action == 'resubscribe':
                            mode = msg.get('mode')
                            if session and mode in self.config.modes:
                                session.subscribed_modes.add(mode)
                                logger.info(f"🔔 {session.uuid} resubscribed: {mode}")

                        else:
                            logger.warning(f"❓ Unknown action: {action}")

                    except json.JSONDecodeError as e:
                        logger.error(f"❌ JSON decode error: {e}")

                # === 二进制消息：视频帧 ===
                elif isinstance(message, bytes):
                    if not session:
                        logger.warning("❌ Frame received before registration")
                        continue

                    if len(message) < 36:
                        logger.warning("❌ Binary message too short")
                        continue

                    # 解析：[36 字节 UUID][JPEG 数据]
                    uuid = message[:36].decode('ascii', errors='ignore').strip()
                    frame_bytes = message[36:]

                    if uuid != session.uuid:
                        logger.warning(f"⚠️ UUID mismatch: {uuid} != {session.uuid}")
                        continue

                    session.update_activity()

                    # 解码帧
                    import cv2, numpy as np
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

                    # 执行分析流水线
                    result = session.process_frame(frame)

                    # 推送 JSON 状态
                    if result['status'] and session.output_json:
                        await self._send_to_client(websocket, {
                            'uuid': uuid,
                            'type': 'status',
                            'data': result['status']
                        })

                    # 推送渲染图像
                    if result['renders'] and session.output_image:
                        for mode, b64_frame in result['renders'].items():
                            await self._send_to_client(websocket, {
                                'uuid': uuid,
                                'type': 'render',
                                'mode': mode,
                                'timestamp': result['status'].get('timestamp'),
                                'frame': b64_frame
                            })

                else:
                    logger.warning(f"❓ Unknown message type: {type(message)}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"🔌 Connection closed: {session.uuid if session else 'unknown'} (code={e.code})")
        except Exception as e:
            logger.error(f"💥 Handler error:", exc_info=True)
        finally:
            if session:
                async with self._lock:
                    await self._remove_session(session.uuid)

    async def run(self):
        """启动服务"""
        logger.info(f"🚀 Starting Shiroha Server at ws://{self.config.host}:{self.config.port}")
        logger.info(f"   Modes: {self.config.modes}")
        logger.info(f"   Headless: {'opencv-contrib-python-headless' in sys.modules}")

        # 启动清理任务
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
                logger.info("✅ Server RUNNING! Press Ctrl+C to stop.")
                while self._running:
                    await asyncio.sleep(1)
        finally:
            self._running = False
            cleanup_task.cancel()
            logger.info("🛑 Server stopped")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Config file path')
    args = parser.parse_args()

    server = ShirohaServer(config_path=args.config)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())