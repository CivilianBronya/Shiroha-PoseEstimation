# -*- coding: utf-8 -*-
import cv2
import threading
import time
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from server.shm_manager import ShmManager

app = Flask(__name__)
CORS(app)

# 初始化共享内存
shm_node = ShmManager(name="shiroha_frame", shape=(480, 640, 3))
frame_buffer = shm_node.create()


def camera_worker():
    """持续将摄像头帧写入共享内存"""
    # TODO：将摄像头选择作为config写入
    cap = cv2.VideoCapture(0) # 摄像头选择
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("摄像头采集已就绪...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # 确保尺寸匹配
            if frame.shape != (480, 640, 3):
                frame = cv2.resize(frame, (640, 480))

            frame_buffer[:] = frame[:]
            time.sleep(0.01)  # 约 60-100 FPS
    finally:
        cap.release()


# flask_app.py 中的 rtc_negotiate 修改
@app.route('/rtc_negotiate', methods=['POST'])
def rtc_negotiate():
    try:
        # 增加超时时间到 15 秒
        # 确保使用的是 127.0.0.1 避免某些系统下 localhost 解析慢
        response = requests.post(
            "http://127.0.0.1:8888/offer",
            json=request.json,
            timeout=15
        )
        return jsonify(response.json())
    except requests.exceptions.Timeout:
        return jsonify({"error": "处理端响应超时，请检查 rtc_main 是否卡死"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # 启动摄像头线程
    t = threading.Thread(target=camera_worker, daemon=True)
    t.start()

    # 🚀 注意：debug=False 极其重要，防止双重初始化共享内存
    app.run(host='0.0.0.0', port=5000, debug=False)