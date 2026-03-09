# 人体姿态与摔倒检测系统

本项目是一个实验性系统，旨在利用计算机视觉技术进行实时人体姿态估计和摔倒检测。它利用 MediaPipe 和 DLib 等库进行关键点检测，并采用基于状态机的算法进行摔倒检测。

## 功能特性

*   实时人体姿态估计。
*   实时头部姿态估计。
*   使用自定义求解器对骨骼数据进行平滑处理。
*   基于状态机的摔倒检测算法 (站立 -> 失衡 -> 摔倒 -> 在地)。
*   渲染骨架的火柴人图形表示。
*   通过 WebSocket 实时流式传输处理后的视频帧。

## 环境要求

*   Python 3.x
*   Conda (Anaconda 或 Miniconda)

若需要websocket，请选择websocket分支进行拉取或克隆，目前该分支为测试阶段，且走向是提供附属功能接入websocket后端
## 安装步骤

1.  **克隆仓库:**
    ```bash
    git clone <repository_url>
    ```

2.  **创建 Conda 环境:**
    ```bash
    conda create -n fall_detection python=3.x # 将 '3.x' 替换为您偏好的 Python 版本 (例如 3.9, 3.10)
    conda activate fall_detection
    ```

3.  **安装依赖项:**
    *(假设您有一个 `requirements.txt` 文件列出了依赖项，如 opencv-python, mediapipe, dlib, websockets 等。)*
    ```bash
    pip install -r requirements.txt
    ```
    *(或者根据需要单独安装。)*

4.  **安装 DLib:**
    DLib 有时需要编译。请遵循您操作系统对应的官方说明：
    *   **Windows/Linux:** `pip install dlib`
    *   **macOS:** `brew install dlib` 然后 `pip install dlib` (或者直接 `pip install dlib` 可能也行)。

5.  **安装 MediaPipe:**
    ```bash
    pip install mediapipe
    ```

6.  **安装 OpenCV:**
    ```bash
    pip install opencv-python
    ```

7.  **安装 WebSockets:**
    ```bash
    pip install websockets
    ```

8.  **(可选) 安装其他依赖项:**
    确保 `requirements.txt` 中列出的所有其他包均已通过 `pip` 安装。

## 使用方法

本项目需要启动两个组件：WebSocket 后端服务器和主检测程序。

1.  **启动 WebSocket 后端服务器:**
    打开一个新的终端或命令行窗口，激活您的 Conda 环境，并导航至项目根目录，然后运行：
    ```bash
    conda activate fall_detection # 激活环境
    cd path/to/PoseEstimation     # 替换为您的项目实际路径
    python -m backend.websocket_backend
    ```
    服务器将启动在 `ws://localhost:8765`。

2.  **启动主检测程序:**
    在**另一个新的**终端或命令行窗口中，重复激活环境并导航至项目根目录的步骤，然后运行：
    ```bash
    python main.py
    ```
    程序将打开摄像头，开始姿态估计和摔倒检测，并将处理后的视频流发送到 WebSocket 服务器。

**按 ESC 键在 `main.py` 窗口中退出应用程序。**

## 终止服务

由于当前系统需手动启动，终止时也需要分别关闭两个进程。

1.  **终止 `main.py`:**
    在运行 `main.py` 的终端窗口中，按 `Ctrl+C` 即可终止程序。

2.  **终止 WebSocket 后端服务器:**
    在 Windows 上，可以通过以下命令查找并终止占用 `8765` 端口的进程：
    ```bash
    netstat -ano | findstr :8765
    ```
    找到结果中的 PID (进程标识符)，然后执行：
    ```bash
    taskkill /PID <PID_NUMBER> /F
    ```
    (请将 `<PID_NUMBER>` 替换为上一条命令查找到的实际数字)。

## 项目状态
**本项目目前处于实验阶段。**

**算法和参数可能需要在不同条件下进行调整以达到最佳性能。**

## 贡献
**本项目为开源项目，欢迎为优化和改进做出贡献。请随时提交 Issue 或 Pull Request。**