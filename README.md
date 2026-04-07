# 人体姿态与摔倒检测系统

本项目是一个实验性系统，旨在利用计算机视觉技术进行实时人体姿态估计和摔倒检测。它利用 MediaPipe 和 DLib 等库进行关键点检测，并采用基于状态机的算法进行摔倒检测，通过 WebRTC 技术实现低延迟的浏览器端实时预览。

## 功能特性

* 实时人体姿态估计：基于 MediaPipe 的高精度骨骼关键点检测。
* 实时头部姿态估计：利用 DLib 进行面部特征点定位与姿态解算。
* 数据平滑处理：使用自定义求解器对骨骼数据进行平滑，减少抖动。
* 摔倒检测算法：基于有限状态机 (FSM) 的逻辑判断 (站立 -> 失衡 -> 摔倒 -> 在地)。
* 可视化渲染：在视频流上实时渲染骨架火柴人图形。
* WebRTC 流媒体传输：支持通过浏览器实时查看处理后的视频流。

## 环境要求

*   操作系统: Windows / Linux / macOS
*   Python: 3.x (推荐 3.8 或更高版本)
*   包管理器: Conda (Anaconda 或 Miniconda)
*   浏览器: 支持 WebRTC 的现代浏览器 (Chrome, Edge, Firefox 等)

## 安装步骤

1.  **克隆仓库:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    # 确保你在 rtcweb 分支
    git checkout rtcweb
    ```

2.  **创建 Conda 环境:**
* Conda是目前必要的环境，除非有更好的办法，本项目为 [Conda Python环境 3.12]
    ```bash
    conda create -n fall_detection python=3.9 # 将 '3.9'以上 替换为您偏好的 Python 版本 (例如 3.9, 3.10)
    conda activate fall_detection
    ```

3.  **安装依赖项:**
    *(假设您有一个 `requirements.txt` 文件列出了依赖项，如 opencv-python, mediapipe, dlib 等。)*
    ```bash
    pip install -r requirements.txt
    ```
    *(或者根据需要单独安装。)*

4.  **安装 DLib:**
* 如果您会安装Conda环境，其实不需要看安装Dlib，但您坚持不需要Conda环境
* 那么DLib 有时需要编译。请遵循您操作系统对应的官方说明，而'我们'只能祝您好运：
    *   **Windows/Linux:** `pip install dlib`
    *   **macOS:** `brew install dlib` 然后 `pip install dlib` (或者直接 `pip install dlib` 可能也行)。

    
5. **(可选) 安装其他依赖项:**
    确保 `requirements.txt` 中列出的所有其他包均已通过 `pip` 安装。


## ！！！注意，该分支为 WebRTC 模式！！！
*(该项目会有更改模块或包的行动，或部分功能重构)*

由于架构调整，本分支不再使用单一的 main.py，而是采用 客户端-服务端 分离的启动模式。请严格按照以下步骤操作：

* 第一步：启动客户端进程

    在终端中运行客户端脚本，它将负责视频采集和初步处理
    ```bash
    python rtc_client.py
    ```
* 第二步：启动主服务进程

    打开一个新的终端窗口（保持第一个窗口运行），激活环境后运行主服务脚本：
    ```bash
    python rtc_main.py
    ```
* 第三步：访问网页端

    在浏览器中打开项目目录下的 demo_rtc_client.html 文件。 
    你可以直接拖拽文件到浏览器，或使用 file:// 协议打开。

* 初始化连接：
    等待几秒钟让后端服务准备就绪。
    点击网页 右上角 的 “初始化所有终端” 按钮。
    如果一切正常，你将看到实时视频流以及叠加的姿态检测结果。

    (*提示： 如果点击按钮后无反应，请检查控制台是否有报错，并确保 rtc_client.py 和 rtc_main.py 都在正常运行且未被防火墙拦截。*)


   **按 ESC 键退出应用程序。**

## 项目状态
**本项目目前处于实验阶段。**

**算法和参数可能需要在不同条件下进行调整以达到最佳性能。**

## 贡献
**本项目为开源项目，欢迎为优化和改进做出贡献。请随时提交 Issue 或 Pull Request。**
