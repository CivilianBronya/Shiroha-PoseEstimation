# -*- coding: utf-8 -*-
"""
HTTP MJPEG Stream Server
浏览器用 <img src="http://host:8766/stream/{uuid}?mode=fall_detector"> 直接显示
"""
import asyncio
import time
from aiohttp import web
import logging
from typing import Dict, Optional
from .frame_buffer import FrameBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mjpeg_server")


class MjpegServer:
    """
    MJPEG 流服务器
    与 WebSocket 服务共享，通过 push_frame() 接收渲染帧
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8766):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.frame_buffers: Dict[str, FrameBuffer] = {}  # uuid -> FrameBuffer
        self._lock = asyncio.Lock()
        self._setup_routes()

    def _setup_routes(self):
        """设置 HTTP 路由"""
        self.app.router.add_get('/stream/{uuid}', self.handle_stream)
        self.app.router.add_get('/frame/{uuid}/{mode}', self.handle_single_frame)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/modes/{uuid}', self.handle_list_modes)

    async def register_buffer(self, uuid: str):
        """为 UUID 注册帧缓冲区"""
        async with self._lock:
            if uuid not in self.frame_buffers:
                self.frame_buffers[uuid] = FrameBuffer(max_age_sec=10.0)
                logger.info(f"📦 FrameBuffer registered: {uuid}")

    async def unregister_buffer(self, uuid: str):
        """移除帧缓冲区"""
        async with self._lock:
            if uuid in self.frame_buffers:
                self.frame_buffers[uuid].clear()
                del self.frame_buffers[uuid]
                logger.info(f"📦 FrameBuffer removed: {uuid}")

    async def push_frame(self, uuid: str, mode: str, jpeg_bytes: bytes):
        """
        从 WebSocket 服务推送渲染帧到缓冲区
        供 MJPEG 流读取
        """
        async with self._lock:
            if uuid not in self.frame_buffers:
                await self.register_buffer(uuid)
            self.frame_buffers[uuid].update(mode, jpeg_bytes)

    async def handle_stream(self, request: web.Request) -> web.StreamResponse:
        """
        MJPEG 流处理器
        GET /stream/{uuid}?mode=fall_detector
        """
        uuid = request.match_info['uuid']
        mode = request.query.get('mode', 'fall_detector')

        logger.info(f"📺 MJPEG stream started: uuid={uuid[:8]}..., mode={mode}")

        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
            }
        )

        await response.prepare(request)

        # 获取/创建帧缓冲区
        async with self._lock:
            if uuid not in self.frame_buffers:
                await self.register_buffer(uuid)
            buffer = self.frame_buffers[uuid]

        last_send_time = 0
        frame_count = 0
        max_fps = 30  # 限流避免浏览器卡死

        try:
            while True:
                # 获取最新帧
                jpeg_bytes = buffer.get(mode)

                if jpeg_bytes:
                    # MJPEG 帧格式: --frame\r\nContent-Type: image/jpeg\r\n\r\n[JPEG 数据]\r\n
                    frame_header = (
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n'
                            b'Content-Length: ' + str(len(jpeg_bytes)).encode() + b'\r\n\r\n'
                    )
                    frame_footer = b'\r\n'

                    await response.write(frame_header + jpeg_bytes + frame_footer)
                    frame_count += 1

                    # 限流：控制最大 FPS
                    now = time.time()
                    min_interval = 1.0 / max_fps
                    elapsed = now - last_send_time
                    if elapsed < min_interval:
                        await asyncio.sleep(min_interval - elapsed)
                    last_send_time = time.time()

                    # 每 100 帧记录日志
                    if frame_count % 100 == 0:
                        logger.debug(f"📊 Stream stats: {frame_count} frames sent for {mode}")
                else:
                    # 无帧时短暂等待
                    await asyncio.sleep(0.05)

        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError,
                ConnectionAbortedError):
            logger.info(f"📺 MJPEG stream closed: uuid={uuid[:8]}..., mode={mode}")
        except Exception as e:
            logger.error(f"❌ Stream error: {e}", exc_info=True)
        finally:
            await response.write_eof()

        return response

    async def handle_single_frame(self, request: web.Request) -> web.Response:
        """
        获取单帧（用于轮询/调试模式）
        GET /frame/{uuid}/{mode}
        """
        uuid = request.match_info['uuid']
        mode = request.match_info['mode']

        async with self._lock:
            buffer = self.frame_buffers.get(uuid)

        if not buffer:
            return web.Response(status=404, text='Session not found')

        jpeg_bytes = buffer.get(mode)
        if not jpeg_bytes:
            return web.Response(status=204, text='No frame available')

        return web.Response(
            body=jpeg_bytes,
            content_type='image/jpeg',
            headers={'Cache-Control': 'no-cache'}
        )

    async def handle_list_modes(self, request: web.Request) -> web.Response:
        """
        列出 UUID 可用的模式
        GET /modes/{uuid}
        """
        uuid = request.match_info['uuid']

        async with self._lock:
            buffer = self.frame_buffers.get(uuid)

        if not buffer:
            return web.json_response({'error': 'Session not found'}, status=404)

        available = list(buffer.get_latest().keys())
        return web.json_response({
            'uuid': uuid,
            'available_modes': available
        })

    async def handle_health(self, request: web.Request) -> web.Response:
        """健康检查"""
        async with self._lock:
            active_sessions = len(self.frame_buffers)
        return web.json_response({
            'status': 'ok',
            'active_sessions': active_sessions,
            'service': 'mjpeg-server',
            'port': self.port
        })

    async def run(self):
        """启动服务"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"🚀 MJPEG server running at http://{self.host}:{self.port}")

        # 保持运行
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            await runner.cleanup()
            logger.info("🛑 MJPEG server stopped")