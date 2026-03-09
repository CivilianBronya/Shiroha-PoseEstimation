# -*- coding: utf-8 -*-
import time
import cv2
import numpy as np
import logging
from typing import Dict, Optional, List

# 导入原有核心模块（复用无需修改）
from pose.body_pose import BodyPose
from face.head_pose import HeadPose
from rig.skeleton import SkeletonSolver
from rig.face_solver import FaceSolver, MODE_SINGLE, MODE_MULTI
from analysis.fall_detector import FallDetector
from render.single_stickman_renderer import SingleStickmanRenderer
from render.multi_stickman_renderer import MultiStickmanRenderer
from render.fall_detector_renderer import FallDetectorRenderer
from render.face_recognition_renderer import FaceRecognitionRenderer
from render.intrusion_detection_renderer import IntrusionDetectionRenderer
from render.loitering_detection_renderer import LoiteringDetectionRenderer
from render.static_detection_renderer import StaticDetectionRenderer
from render.vigorous_activity_renderer import VigorousActivityRenderer
from render.activity_level_renderer import ActivityLevelRenderer
from .output_encoder import encode_frame_to_base64

logger = logging.getLogger(__name__)


class Session:
    """
    单 UUID 分析会话：封装原 main.py 的核心分析流水线
    每个会话独立维护组件状态，实现多租户隔离
    """

    # 模式名 → 渲染器类映射（按需懒加载）
    RENDERER_MAP = {
        'fall_detector': FallDetectorRenderer,
        'pose_monitoring': MultiStickmanRenderer,
        'face_recognition': FaceRecognitionRenderer,
        'intrusion_detection': IntrusionDetectionRenderer,
        'emotion_detection': FaceRecognitionRenderer,  # 复用单人模式
        'loitering_detection': LoiteringDetectionRenderer,
        'static_detection': StaticDetectionRenderer,
        'vigorous_activity': VigorousActivityRenderer,
        'activity_level': ActivityLevelRenderer,
        'motion_capture': None,  # 原始帧，无需渲染器
    }

    def __init__(self, uuid: str, config: dict, server_config):
        self.uuid = uuid
        self.config = config
        self.server_config = server_config
        self.created_at = time.time()
        self.last_active = time.time()
        self.websocket = None  # 由 server 注入

        # 订阅的模式列表
        self.subscribed_modes = set(config.get('modes', server_config.modes))
        self.output_image = config.get('output_format', {}).get('image', True)
        self.output_json = config.get('output_format', {}).get('json', True)

        # === 初始化分析组件（与原 main.py 一致）===
        self.body_single = BodyPose(
            model_path=server_config.pose_model_path,
            num_poses=server_config.pose['single']['num_poses']
        )
        self.body_multi = BodyPose(
            model_path=server_config.pose_model_path,
            num_poses=server_config.pose['multi']['num_poses']
        )
        self.head_pose = HeadPose()  # 依赖 dlib 模型
        self.skeleton_solver = SkeletonSolver(
            filter_alpha=server_config.skeleton.get('filter_alpha', 0.7)
        )

        # 人脸检测（Haar + dlib 平滑）
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.face_solver_multi = FaceSolver(
            filter_alpha=server_config.face.get('filter_alpha', 0.7),
            min_area=server_config.face.get('min_area', 2000),
            mode=MODE_MULTI
        )
        self.face_solver_single = FaceSolver(
            filter_alpha=server_config.face.get('filter_alpha', 0.7),
            min_area=server_config.face.get('min_area', 2000),
            mode=MODE_SINGLE
        )

        # 摔倒检测
        self.fall_detector = FallDetector(
            ground_threshold_sec=server_config.fall_detector.get('ground_threshold_sec', 4.5)
        )

        # 渲染器懒加载字典
        self._renderers = {}
        self._single_renderer = SingleStickmanRenderer()  # fall_detector 专用

        logger.info(f"✅ Session initialized: {uuid}, modes={self.subscribed_modes}")

    def _get_renderer(self, mode: str):
        """懒加载渲染器"""
        if mode in self._renderers:
            return self._renderers[mode]

        renderer_cls = self.RENDERER_MAP.get(mode)
        if not renderer_cls:
            return None

        # 特殊初始化逻辑
        if mode == 'fall_detector':
            renderer = renderer_cls(self.fall_detector)
        elif mode in ('face_recognition', 'emotion_detection'):
            solver = self.face_solver_multi if mode == 'face_recognition' else self.face_solver_single
            renderer = renderer_cls(solver)
        elif mode == 'loitering_detection':
            cfg = self.server_config.detection.get('loitering', {})
            renderer = renderer_cls(
                alert_duration=cfg.get('alert_duration', 15),
                cycle_length=cfg.get('cycle_length', 90),
                alert_threshold=cfg.get('alert_threshold', 10)
            )
        elif mode == 'static_detection':
            cfg = self.server_config.detection.get('static', {})
            renderer = renderer_cls(
                history_length=cfg.get('history_length', 30),
                movement_threshold=cfg.get('movement_threshold', 15.0)
            )
        elif mode == 'vigorous_activity':
            cfg = self.server_config.detection.get('vigorous_activity', {})
            renderer = renderer_cls(
                activity_threshold=cfg.get('activity_threshold', 30.0)
            )
        elif mode == 'activity_level':
            cfg = self.server_config.detection.get('activity_level', {})
            renderer = renderer_cls(
                low_threshold=cfg.get('low_threshold', 15.0),
                high_threshold=cfg.get('high_threshold', 35.0)
            )
        else:
            renderer = renderer_cls()

        self._renderers[mode] = renderer
        return renderer

    def update_activity(self):
        """更新最后活跃时间（用于超时清理）"""
        self.last_active = time.time()

    def is_expired(self, timeout_sec: float = None) -> bool:
        """检查会话是否超时"""
        timeout = timeout_sec or self.server_config.session_timeout
        return (time.time() - self.last_active) > timeout

    def process_frame(self, frame_bgr: np.ndarray) -> Dict:
        """
        核心分析流水线：与原 main.py 逻辑完全一致
        :param frame_bgr: OpenCV BGR 帧
        :return: {'status': {...}, 'renders': {'mode': base64_str}}
        """
        result = {'status': {}, 'renders': {}}
        h, w = frame_bgr.shape[:2]

        # === 1. 单人姿态检测（用于摔倒/活动分析）===
        raw_single = self.body_single.detect(frame_bgr)
        body_pts_list = None
        skeleton = None

        if raw_single and isinstance(raw_single, dict):
            body_pts_list = raw_single.get('landmark_points')
            if body_pts_list:
                body_dict = {i: pt for i, pt in enumerate(body_pts_list)}
                head_rot = self.head_pose.detect(frame_bgr)
                skeleton = self.skeleton_solver.solve(body_dict, head_rot)
                # 更新摔倒检测状态
                if skeleton:
                    self.fall_detector.update(skeleton)

        # === 2. 多人姿态检测（用于监控类渲染）===
        raw_multi = self.body_multi.detect(frame_bgr)
        multi_body_data = []
        if raw_multi and 'people' in raw_multi:
            for person in raw_multi['people']:
                if 'landmark_points' in person:
                    multi_body_data.append(person['landmark_points'])

        # === 3. 人脸检测（Haar Cascade）===
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces_raw = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        faces_for_render = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in faces_raw]

        # === 4. 构建 JSON 状态输出 ===
        if self.output_json:
            result['status'] = {
                'uuid': self.uuid,
                'timestamp': time.time(),
                'pose_count': len(multi_body_data),
                'face_count': len(faces_for_render),
                # 摔倒相关
                'fall_detected': self.fall_detector.get_fall_status() if skeleton else False,
                'fall_risk_score': self.fall_detector.get_fall_risk_score() if skeleton else 0,
                'fall_state': self.fall_detector.get_state_name() if skeleton else 'NO_SKELETON',
                # 可扩展：activity_level, intrusion_alert 等
            }

        # === 5. 按需渲染图像（仅推送订阅的模式）===
        if self.output_image:
            for mode in self.subscribed_modes:
                try:
                    render_frame = None

                    if mode == 'motion_capture':
                        # 原始帧 + FPS 水印
                        render_frame = frame_bgr.copy()
                        cv2.putText(render_frame, f"UUID:{self.uuid[:8]}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    elif mode == 'fall_detector':
                        debug_frame = frame_bgr.copy()
                        if body_pts_list and skeleton:
                            debug_frame = self._single_renderer.draw(debug_frame, body_pts_list, skeleton)
                        renderer = self._get_renderer('fall_detector')
                        if renderer:
                            debug_frame = renderer.draw(debug_frame, skeleton)
                        render_frame = debug_frame

                    elif mode == 'pose_monitoring':
                        debug_frame = frame_bgr.copy()
                        renderer = self._get_renderer('pose_monitoring')
                        if renderer and multi_body_data:
                            debug_frame = renderer.draw(debug_frame, multi_body_data)
                        render_frame = debug_frame

                    elif mode == 'face_recognition':
                        debug_frame = frame_bgr.copy()
                        renderer = self._get_renderer('face_recognition')
                        if renderer:
                            debug_frame = renderer.draw(debug_frame, faces_for_render)
                        render_frame = debug_frame

                    elif mode == 'emotion_detection':
                        debug_frame = frame_bgr.copy()
                        renderer = self._get_renderer('emotion_detection')
                        if renderer:
                            debug_frame = renderer.draw(debug_frame, faces_for_render)
                        render_frame = debug_frame

                    elif mode == 'intrusion_detection':
                        debug_frame = frame_bgr.copy()
                        renderer = self._get_renderer('intrusion_detection')
                        if renderer:
                            debug_frame = renderer.draw(debug_frame, multi_body_data)
                        render_frame = debug_frame

                    elif mode == 'loitering_detection':
                        debug_frame = frame_bgr.copy()
                        renderer = self._get_renderer('loitering_detection')
                        if renderer:
                            debug_frame = renderer.draw(debug_frame, multi_body_data)
                        render_frame = debug_frame

                    elif mode == 'static_detection':
                        debug_frame = frame_bgr.copy()
                        renderer = self._get_renderer('static_detection')
                        if renderer:
                            debug_frame = renderer.draw(debug_frame, multi_body_data)
                        render_frame = debug_frame

                    elif mode == 'vigorous_activity':
                        debug_frame = frame_bgr.copy()
                        renderer = self._get_renderer('vigorous_activity')
                        if renderer and body_pts_list:
                            debug_frame = renderer.draw(debug_frame, body_pts_list)
                        render_frame = debug_frame

                    elif mode == 'activity_level':
                        debug_frame = frame_bgr.copy()
                        renderer = self._get_renderer('activity_level')
                        if renderer and body_pts_list:
                            debug_frame = renderer.draw(debug_frame, body_pts_list)
                        render_frame = debug_frame

                    # 编码为 base64
                    if render_frame is not None:
                        b64 = encode_frame_to_base64(
                            render_frame,
                            quality=self.server_config.jpeg_quality
                        )
                        if b64:
                            result['renders'][mode] = b64

                except Exception as e:
                    logger.warning(f"Render error [{mode}] for {self.uuid}: {e}")
                    continue

        return result

    def cleanup(self):
        """会话销毁时释放资源"""
        logger.info(f"🧹 Cleaning up session: {self.uuid}")
        # MediaPipe 显式释放（如支持）
        if hasattr(self.body_single, 'detector') and hasattr(self.body_single.detector, 'close'):
            self.body_single.detector.close()
        if hasattr(self.body_multi, 'detector') and hasattr(self.body_multi.detector, 'close'):
            self.body_multi.detector.close()
        self._renderers.clear()