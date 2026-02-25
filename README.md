# 人体姿态与摔倒检测系统

本项目是一个实验性系统，旨在利用计算机视觉技术进行实时人体姿态估计和摔倒检测。它利用 MediaPipe 和 DLib 等库进行关键点检测，并采用基于状态机的算法进行摔倒检测。

## 功能特性

*   实时人体姿态估计。
*   实时头部姿态估计。
*   使用自定义求解器对骨骼数据进行平滑处理。
*   基于状态机的摔倒检测算法 (站立 -> 失衡 -> 摔倒 -> 在地)。
*   渲染骨架的火柴人图形表示。
*   JSON 格式输出，用于后续处理。

## 环境要求

*   Python 3.x
*   Conda (Anaconda 或 Miniconda)

## 安装步骤

1.  **克隆仓库:**
    ```bash
    git clone <your_repository_url>
    cd <repository_name>
    ```

2.  **创建 Conda 环境:**
    ```bash
    conda create -n fall_detection python=3.x # 将 '3.x' 替换为您偏好的 Python 版本 (例如 3.9, 3.10)
    conda activate fall_detection
    ```

3.  **安装依赖项:**
    *(假设您有一个 `requirements.txt` 文件列出了依赖项，如 opencv-python, mediapipe, dlib 等。)*
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

7.  **(可选) 安装其他依赖项:**
    确保 `requirements.txt` 中列出的所有其他包均已通过 `pip` 安装。

## 使用方法

*(提供运行主脚本的说明)*

```bash
python main.py
```
   **按 ESC 键退出应用程序。**

## 项目状态
**本项目目前处于实验阶段。**

**算法和参数可能需要在不同条件下进行调整以达到最佳性能。**

## 贡献
**本项目为开源项目，欢迎为优化和改进做出贡献。请随时提交 Issue 或 Pull Request。**
