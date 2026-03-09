import websockets
import uuid
import json
from datetime import datetime

connected_clients = {}


async def handle_client(websocket):
    """
    处理单个 WebSocket 客户端连接。
    注意：新版本websockets库的handler只接收websocket对象
    """
    # 为新连接生成一个唯一的会话ID
    session_id = str(uuid.uuid4())

    # 将新连接添加到全局集合中
    connected_clients[session_id] = {
        'websocket': websocket,
        'join_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    print(f"[{datetime.now()}] New client connected. Session ID: {session_id}. Total clients: {len(connected_clients)}")

    try:
        # 向客户端发送欢迎消息，包含其会话ID
        welcome_msg = {
            "type": "welcome",
            "session_id": session_id,
            "message": f"Connected successfully! Your session ID is {session_id}"
        }
        await websocket.send(json.dumps(welcome_msg))

        # 等待客户端关闭连接
        await websocket.wait_closed()

    except websockets.exceptions.ConnectionClosedOK:
        print(f"[{datetime.now()}] Client {session_id} disconnected gracefully.")
    except Exception as e:
        print(f"[{datetime.now()}] Error handling client {session_id}: {e}")
    finally:
        # 客户端断开，从集合中移除
        if session_id in connected_clients:
            del connected_clients[session_id]
        print(f"[{datetime.now()}] Session {session_id} cleaned up. Total clients: {len(connected_clients)}")


async def broadcast_frame_to_all_clients(encoded_frame_bytes):
    """
    将一帧编码后的图像数据广播给所有已连接的客户端。
    """
    if not connected_clients:
        return

    message = {
        "type": "video_frame",
        "timestamp": datetime.now().isoformat()
    }

    to_remove = set()
    for session_id, client_info in connected_clients.items():
        try:
            ws = client_info['websocket']
            await ws.send(json.dumps(message))
            await ws.send(encoded_frame_bytes)
        except websockets.exceptions.ConnectionClosed:
            print(f"Client {session_id} seems to have disconnected during broadcast.")
            to_remove.add(session_id)
        except Exception as e:
            print(f"Error broadcasting to {session_id}: {e}")
            to_remove.add(session_id)

    for session_id in to_remove:
        if session_id in connected_clients:
            del connected_clients[session_id]

def get_connected_clients_count():
    return len(connected_clients)