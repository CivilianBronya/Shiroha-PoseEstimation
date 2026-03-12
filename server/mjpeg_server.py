# -*- coding: utf-8 -*-
import asyncio
import time
import logging
from aiohttp import web
from typing import Dict, Optional

logger = logging.getLogger("mjpeg_server")


class FrameBuffer:
    """高效帧缓冲区：使用 Event 通知新帧到达"""

    def __init__(self, max_age_sec: float = 5.0):
        self.max_age_sec = max_age_sec
        self.frames: Dict[str, bytes] = {}
        self.timestamps: Dict[str, float] = {}
        # 🚀 信号灯：一旦有新帧存入，立即触发等待的协程
        self.new_frame_event = asyncio.Event()

    def update(self, mode: str, jpeg_bytes: bytes):
        self.frames[mode] = jpeg_bytes
        self.timestamps[mode] = time.time()
        # 触发事件唤醒所有 handle_stream 的循环
        self.new_frame_event.set()
        # 立即重置，准备迎接下一帧
        self.new_frame_event.clear()

    def get(self, mode: str) -> Optional[bytes]:
        now = time.time()
        ts = self.timestamps.get(mode, 0)
        if now - ts > self.max_age_sec:
            return None
        return self.frames.get(mode)


class MjpegServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8766):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.frame_buffers: Dict[str, FrameBuffer] = {}
        self._setup_routes()

    def _setup_routes(self):
        """定义 HTTP 路由"""
        self.app.router.add_get('/stream/{uuid}', self.handle_stream)
        self.app.router.add_get('/health', self.handle_health)
        logger.info("✅ MJPEG routes configured")

    async def register_buffer(self, uuid: str):
        if uuid not in self.frame_buffers:
            self.frame_buffers[uuid] = FrameBuffer()
            logger.info(f"📦 Registered MJPEG buffer: {uuid}")

    async def unregister_buffer(self, uuid: str):
        if uuid in self.frame_buffers:
            del self.frame_buffers[uuid]
            logger.info(f"🗑️ Unregistered MJPEG buffer: {uuid}")

    def push_frame(self, uuid: str, mode: str, jpeg_bytes: bytes):
        """同步非阻塞推送，由 ws_server 调用"""
        if uuid in self.frame_buffers:
            self.frame_buffers[uuid].update(mode, jpeg_bytes)

    async def handle_stream(self, request: web.Request) -> web.StreamResponse:
        """MJPEG 流处理器：实时推送渲染帧"""
        uuid = request.match_info['uuid']
        mode = request.query.get('mode', 'fall_detector')

        if uuid not in self.frame_buffers:
            await self.register_buffer(uuid)

        buffer = self.frame_buffers[uuid]

        response = web.StreamResponse(status=200, headers={
            'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
        })
        await response.prepare(request)

        logger.info(f"📺 MJPEG Stream Started: {uuid[:8]}... Mode: {mode}")

        try:
            while True:
                # 🚀 核心优化：阻塞等待直到新帧产生，0% CPU 占用
                await buffer.new_frame_event.wait()

                jpeg_bytes = buffer.get(mode)
                if jpeg_bytes:
                    header = (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n'
                              b'Content-Length: ' + str(len(jpeg_bytes)).encode() + b'\r\n\r\n')
                    await response.write(header + jpeg_bytes + b'\r\n')
                    # 🚀 强制将数据推入网络套接字，减少延迟
                    await response.drain()

                # 限制最大 FPS 约为 33，防止浏览器压力过大
                await asyncio.sleep(0.03)

        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            logger.info(f"📺 MJPEG Stream Stopped: {uuid[:8]}...")
        except Exception as e:
            logger.error(f"❌ MJPEG Stream Error: {e}")
        finally:
            return response

    async def handle_health(self, request: web.Request):
        """健康检查接口"""
        return web.json_response({
            "status": "ok",
            "active_buffers": len(self.frame_buffers),
            "timestamp": time.time()
        })

    async def run(self):
        """启动 aiohttp 服务"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"🚀 MJPEG server running at http://{self.host}:{self.port}")