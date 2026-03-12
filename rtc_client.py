import cv2
import time
import threading
import requests
from flask import Flask, render_template, request, jsonify
from server.shm_manager import ShmManager  # 导入刚才定义的工具类

app = Flask(__name__)

# --- 1. 初始化共享内存 ---
# 确保 shape 与处理端完全一致 (H, W, C)
shm_node = ShmManager(name="shiroha_frame", shape=(480, 640, 3))
frame_buffer = shm_node.create()


# --- 2. 摄像头采集线程 ---
def camera_producer():
    cap = cv2.VideoCapture(0)
    # 强制设置分辨率，确保与共享内存大小匹配
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("📹 摄像头采集线程已启动...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 🚀 零拷贝写入共享内存
            # 使用 [:] 确保是在原有内存空间上覆盖数据
            frame_buffer[:] = frame[:]

            # 控制采集率，避免过度占用 CPU
            time.sleep(0.01)
    finally:
        cap.release()
        shm_node.close()


# 启动采集后台线程
threading.Thread(target=camera_producer, daemon=True).start()


# --- 3. 路由设置 ---

@app.route('/')
def index():
    # 渲染刚才写的 index.html
    return render_template('./server/server_test_demo/demo_rtc_client.html')


@app.route('/rtc_negotiate', methods=['POST'])
def rtc_negotiate():
    """
    信令中转站：前端 Offer -> Flask -> 本项目(8888) -> Flask -> 前端 Answer
    """
    client_offer = request.json
    try:
        # 转发 SDP Offer 给处理端 (rtc_main.py)
        # 注意：这里的 mode 参数也会被带过去
        resp = requests.post("http://localhost:8888/offer", json=client_offer, timeout=5)
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # 建议关闭 debug 模式，因为多线程/多进程在 debug 模式下可能会初始化两次共享内存
    app.run(host='0.0.0.0', port=5000, debug=False)