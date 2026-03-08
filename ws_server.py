# -*- coding: utf-8 -*-
"""
调试服务端 v2 - 支持图像回显 (render 推送)
用途：确认视频帧收发 + 渲染图像推送是否正常
"""
import asyncio
import websockets
import json
import base64
import logging
import sys
import traceback
import time
import cv2
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('server_debug.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger("debug_server_v2")


class DebugSession:
    def __init__(self, uuid: str, config: dict = None):
        self.uuid = uuid
        self.config = config or {}
        self.websocket = None
        self.last_active = time.time()
        self.modes = self.config.get('modes', ['fall_detector'])

    def update_activity(self):
        self.last_active = time.time()


def encode_frame_to_base64(frame_bgr, quality=75) -> str:
    """将 OpenCV 帧编码为 base64 JPEG 字符串"""
    try:
        _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode('ascii')
    except Exception as e:
        logger.error(f"Frame encode error: {e}")
        return None


def add_watermark(frame_bgr, text: str) -> np.ndarray:
    """在帧上添加水印文字（模拟渲染）"""
    overlay = frame_bgr.copy()
    # 添加半透明矩形背景
    cv2.rectangle(overlay, (10, 10), (300, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)
    # 添加文字
    cv2.putText(frame_bgr, text, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame_bgr


async def handler(websocket):
    """WebSocket 处理器 - 支持 status + render 双推送"""
    session = None

    try:
        logger.info("🔗 New connection established")

        async for message in websocket:
            try:
                # === 处理文本消息（控制命令）===
                if isinstance(message, str):
                    logger.debug(f"📥 Received TEXT: {message[:200]}...")
                    msg = json.loads(message)
                    action = msg.get('action')

                    if action == 'register':
                        uuid = msg.get('uuid', 'unknown')
                        config = msg.get('config', {})
                        session = DebugSession(uuid, config)
                        session.websocket = websocket

                        logger.info(f"📋 Registered: uuid={uuid}, modes={session.modes}")

                        # 发送确认
                        await websocket.send(json.dumps({
                            'type': 'ack',
                            'uuid': uuid,
                            'status': 'registered',
                            'server': 'debug_v2'
                        }))

                    elif action == 'ping':
                        await websocket.send(json.dumps({'type': 'pong'}))

                    elif action == 'unsubscribe':
                        mode = msg.get('mode')
                        if session and mode in session.modes:
                            session.modes.remove(mode)
                            logger.info(f"🔕 Unsubscribed: {mode}")

                    else:
                        logger.warning(f"❓ Unknown action: {action}")

                # === 处理二进制消息（视频帧）===
                elif isinstance(message, bytes):
                    if len(message) < 36:
                        logger.warning("❌ Binary message too short (<36 bytes)")
                        continue

                    # 解析 UUID + 帧数据
                    uuid_bytes = message[:36]
                    frame_bytes = message[36:]
                    uuid = uuid_bytes.decode('ascii', errors='ignore').strip()

                    logger.debug(f"🎬 Frame: uuid={uuid}, size={len(frame_bytes)}B")

                    # 验证会话
                    if not session or uuid != session.uuid:
                        logger.warning(f"⚠️ UUID mismatch: {uuid} != {session.uuid if session else 'None'}")
                        continue

                    session.update_activity()

                    # 解码帧
                    try:
                        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
                            raise ValueError("Decode failed")
                    except Exception as e:
                        logger.error(f"❌ Frame decode error: {e}")
                        continue

                    # === 1. 发送 status 消息（原有逻辑）===
                    status_data = {
                        'uuid': uuid,
                        'timestamp': time.time(),
                        'frame_received': True,
                        'frame_size_kb': round(len(frame_bytes) / 1024, 1),
                        'pose_count': 1,  # mock
                        'fall_detected': False,
                        'fall_risk_score': 15,
                        'fall_state': 'STAND',
                        'message': 'Frame processed successfully'
                    }
                    await websocket.send(json.dumps({
                        'uuid': uuid,
                        'type': 'status',
                        'data': status_data
                    }))

                    # === 2. 发送 render 消息（新增！）===
                    # 仅推送客户端订阅的模式
                    for mode in session.modes:
                        try:
                            # 模拟渲染：添加水印 + 边框
                            render_frame = add_watermark(frame.copy(), f"🎯 {mode.upper()}")

                            # 添加模式特定视觉效果
                            if mode == 'fall_detector':
                                cv2.rectangle(render_frame, (50, 100), (200, 250), (0, 0, 255), 3)
                                cv2.putText(render_frame, "FALL ALERT", (55, 120),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            elif mode == 'pose_monitoring':
                                # 画几个模拟骨架点
                                for i in range(0, 100, 20):
                                    cv2.circle(render_frame, (100 + i, 150), 5, (0, 255, 0), -1)
                            elif mode == 'intrusion':
                                cv2.polylines(render_frame, [np.array([[30, 80], [250, 80], [250, 200], [30, 200]])],
                                              True, (255, 0, 255), 2)
                                cv2.putText(render_frame, "ZONE", (40, 95),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                            # 编码为 base64
                            frame_b64 = encode_frame_to_base64(render_frame, quality=70)
                            if frame_b64:
                                await websocket.send(json.dumps({
                                    'uuid': uuid,
                                    'type': 'render',
                                    'mode': mode,
                                    'timestamp': time.time(),
                                    'frame': frame_b64
                                }))
                                logger.debug(f"📤 Sent render: mode={mode}, size={len(frame_b64)}B")
                            else:
                                logger.warning(f"⚠️ Failed to encode render frame for {mode}")

                        except Exception as e:
                            logger.error(f"❌ Render error for mode '{mode}': {e}", exc_info=True)

                else:
                    logger.warning(f"❓ Unknown message type: {type(message)}")

            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON decode error: {e}")
            except Exception as e:
                logger.error(f"❌ Message handling ERROR:", exc_info=True)
                try:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': str(e),
                        'traceback': traceback.format_exc()
                    }))
                except:
                    pass

    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"🔌 Connection closed: uuid={session.uuid if session else 'unknown'}, code={e.code}")
    except Exception as e:
        logger.error(f"💥 Handler CRASHED:", exc_info=True)
    finally:
        if session:
            logger.info(f"🧹 Cleanup session: {session.uuid}")


async def main():
    host = "0.0.0.0"
    port = 8765

    logger.info(f"🚀 Starting DEBUG SERVER V2 at ws://{host}:{port}")
    logger.info(f"   Python: {sys.version}")
    logger.info(f"   OpenCV: {cv2.__version__}")
    logger.info(f"   websockets: {websockets.__version__}")

    try:
        async with websockets.serve(
                handler,
                host,
                port,
                ping_interval=20,
                ping_timeout=10,
                max_size=2 ** 24
        ) as server:
            logger.info("✅ Server RUNNING! Press Ctrl+C to stop.")
            logger.info("📝 Logs: server_debug.log")
            await asyncio.Future()
    except OSError as e:
        logger.error(f"❌ Port bind failed: {e}")
        input("Press Enter to exit...")
    except Exception as e:
        logger.error(f"💥 Startup failed:", exc_info=True)
        input("Press Enter to exit...")


if __name__ == "__main__":
    asyncio.run(main())