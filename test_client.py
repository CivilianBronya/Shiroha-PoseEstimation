# -*- coding: utf-8 -*-
import asyncio
import websockets
import cv2
import uuid
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_client")


async def test_client():
    my_uuid = str(uuid.uuid4())
    uri = "ws://localhost:8765"

    logger.info(f"Connecting to {uri} with UUID: {my_uuid}")

    try:
        async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
            logger.info("✅ Connected!")

            # 1. 注册
            register_msg = {
                "action": "register",
                "uuid": my_uuid,
                "config": {
                    "modes": ["fall_detector"],
                    "output_format": {"image": True, "json": True}
                }
            }
            logger.info(f"Sending register: {register_msg}")
            await ws.send(json.dumps(register_msg))

            # 等待响应（带超时）
            try:
                ack = await asyncio.wait_for(ws.recv(), timeout=5.0)
                logger.info(f"✅ Received ack: {ack}")
            except asyncio.TimeoutError:
                logger.error("❌ Timeout waiting for ack")
                return

            # 2. 打开摄像头推流
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("❌ Cannot open camera")
                return

            logger.info("📷 Camera opened, start sending frames...")
            frame_count = 0

            while frame_count < 30:  # 测试 30 帧后退出
                ret, frame = cap.read()
                if not ret:
                    logger.error("❌ Failed to grab frame")
                    break

                # 编码为 JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                # 构造二进制消息：[UUID 36 字节][JPEG 数据]
                uuid_bytes = my_uuid.ljust(36).encode('ascii')
                message = uuid_bytes + buffer.tobytes()

                await ws.send(message)
                frame_count += 1

                # 接收结果（非阻塞）
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    if isinstance(response, str):
                        msg = json.loads(response)
                        logger.info(f"📊 Received: {msg.get('type')} - {msg.get('data', {})}")
                except asyncio.TimeoutError:
                    pass

                await asyncio.sleep(0.033)  # ~30 FPS

            cap.release()
            logger.info(f"✅ Test completed, {frame_count} frames sent")

    except websockets.exceptions.InvalidStatusCode as e:
        logger.error(f"❌ Connection failed (status {e.status_code}): {e}")
        logger.error("💡 请确认服务端已启动：python server/ws_server_v12.py")
    except ConnectionRefusedError:
        logger.error("❌ Connection refused")
        logger.error("💡 请确认服务端已启动且端口 8765 未被占用")
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(test_client())
    input("Press Enter to exit...")  # 保持窗口查看日志