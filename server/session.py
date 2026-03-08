import time

import cv2

from pose.body_pose import BodyPose
from face.head_pose import HeadPose
from rig.skeleton import SkeletonSolver
from analysis.fall_detector import FallDetector
from render.fall_detector_renderer import FallDetectorRenderer
from render.multi_stickman_renderer import MultiStickmanRenderer



class Session:
    """单个 UUID 的分析会话，隔离状态与组件"""

    def __init__(self, uuid: str, config: dict):
        self.uuid = uuid
        self.config = config
        self.created_at = time.time()
        self.last_active = time.time()

        # 🔑 每个会话独立实例，避免状态污染
        self.body_pose = BodyPose(
            model_path=config['pose_model_path'],
            num_poses=config.get('max_poses', 4)
        )
        self.head_pose = HeadPose()
        self.skeleton_solver = SkeletonSolver(filter_alpha=0.7)
        self.fall_detector = FallDetector()

        # 渲染器按需初始化（节省内存）
        self.renderers = {}
        if 'fall_detector' in config.get('modes', []):
            self.renderers['fall_detector'] = FallDetectorRenderer(self.fall_detector)
        if 'pose_monitoring' in config.get('modes', []):
            self.renderers['pose_monitoring'] = MultiStickmanRenderer()
        # ... 其他 renderer 按需加载

        # 输出订阅配置
        self.output_config = config.get('output_format', {'image': True, 'json': True})
        self.subscribed_modes = set(config.get('modes', []))

    def update_activity(self):
        """更新最后活跃时间"""
        self.last_active = time.time()

    def is_expired(self, timeout_sec: float = 300.0) -> bool:
        """检查会话是否超时"""
        return (time.time() - self.last_active) > timeout_sec

    def process_frame(self, frame_bgr) -> dict:
        """
        处理单帧，返回结构化结果
        :param frame_bgr: numpy array (H,W,3) BGR format
        :return: {'status': {...}, 'renders': {'mode_name': frame_bytes}}
        """
        result = {'status': {}, 'renders': {}}

        # === 1. 姿态检测 ===
        raw_result = self.body_pose.detect(frame_bgr)
        multi_body_data = []
        if raw_result and 'people' in raw_result:
            for person in raw_result['people']:
                if 'landmark_points' in person:
                    multi_body_data.append(person['landmark_points'])

        # === 2. 核心分析（以摔倒检测为例）===
        if multi_body_data:
            # 取第一人做摔倒分析（可扩展多人）
            skeleton = self.skeleton_solver.solve(
                {i: pt for i, pt in enumerate(multi_body_data[0])},
                self.head_pose.detect(frame_bgr)
            )
            if skeleton:
                self.fall_detector.update(skeleton)
                result['status'].update({
                    'fall_detected': self.fall_detector.get_fall_status(),
                    'fall_risk_score': self.fall_detector.get_fall_risk_score(),
                    'fall_state': self.fall_detector.get_state_name(),
                    'skeleton': skeleton  # 可选：返回原始骨架数据
                })

        # === 3. 按需渲染（仅推送订阅的模式）===
        if self.output_config.get('image'):
            for mode_name, renderer in self.renderers.items():
                if mode_name not in self.subscribed_modes:
                    continue
                try:
                    if mode_name == 'fall_detector':
                        debug_frame = renderer.draw(frame_bgr.copy(), skeleton)
                    elif mode_name == 'pose_monitoring':
                        debug_frame = renderer.draw(frame_bgr.copy(), multi_body_data)
                    # ... 其他 renderer 调用

                    # 编码为 JPEG bytes（减少传输体积）
                    _, buffer = cv2.imencode('.jpg', debug_frame,
                                             [cv2.IMWRITE_JPEG_QUALITY, 75])
                    result['renders'][mode_name] = buffer.tobytes()
                except Exception as e:
                    # 渲染失败不影响主流程
                    print(f"Render {mode_name} error: {e}")

        # === 4. 补充状态信息 ===
        if self.output_config.get('json'):
            result['status'].update({
                'uuid': self.uuid,
                'timestamp': time.time(),
                'pose_count': len(multi_body_data),
                # 可扩展：activity_level, intrusion_alert 等
            })

        return result

    def cleanup(self):
        """会话销毁时释放资源"""
        # MediaPipe 等可能需要显式释放
        if hasattr(self.body_pose, 'detector'):
            self.body_pose.detector.close()