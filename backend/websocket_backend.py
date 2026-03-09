import asyncio
import websockets
import queue
from datetime import datetime
from .websocket_handlers import handle_client, broadcast_frame_to_all_clients


async def main():
    """
    主函数，启动 WebSocket 服务器。
    """
    # 启动 WebSocket 服务器，将请求交给 handle_client 处理
    server = await websockets.serve(handle_client, "localhost", 8765)
    print(f"[{datetime.now()}] WebSocket server started on ws://localhost:8765")

    # 传递帧
    frame_queue = queue.Queue(maxsize=2)

    import sys
    sys._frame_queue_for_backend = frame_queue

    async def broadcast_loop():
        while True:
            try:
                # 非阻塞获取帧
                encoded_frame = frame_queue.get_nowait()
                await broadcast_frame_to_all_clients(encoded_frame)
            except queue.Empty:
                pass
            await asyncio.sleep(0.01)

    asyncio.create_task(broadcast_loop())

    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())

def put_frame_for_broadcast(encoded_frame_bytes):
    """
    一个可以从 main.py 调用的函数，将编码好的帧放入队列以供广播。
    """
    import sys
    q = getattr(sys, '_frame_queue_for_backend', None)
    if q:
        try:
            if not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put_nowait(encoded_frame_bytes)
        except queue.Full:
            pass
    else:
        print("Global frame queue not initialized in backend.")


def get_client_count():
    """从main.py获取当前连接的客户端数量"""
    from .websocket_handlers import get_connected_clients_count
    return get_connected_clients_count()