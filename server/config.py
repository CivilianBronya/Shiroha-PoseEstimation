# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ServerConfig:
    """服务端配置（加载自 config.json）"""
    host: str = "0.0.0.0"
    port: int = 8765
    mjpeg_port: int = 8766  # 🔥 新增：MJPEG 服务端口
    session_timeout: float = 300.0
    jpeg_quality: int = 75

    # 模型路径
    pose_model_path: str = "models/pose_landmarker_full.task"
    face_model_path: str = "models/shape_predictor_68_face_landmarks.dat"

    # 组件参数
    camera: Dict[str, Any] = field(default_factory=dict)
    pose: Dict[str, Any] = field(default_factory=dict)
    skeleton: Dict[str, Any] = field(default_factory=dict)
    face: Dict[str, Any] = field(default_factory=dict)
    fall_detector: Dict[str, Any] = field(default_factory=dict)
    detection: Dict[str, Any] = field(default_factory=dict)

    # 可用模式列表
    modes: List[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str = "config.json") -> 'ServerConfig':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls(
            host=data.get('server', {}).get('host', '0.0.0.0'),
            port=data.get('server', {}).get('port', 8765),
            mjpeg_port=data.get('server', {}).get('mjpeg_port', 8766),  # 🔥 新增
            session_timeout=data.get('server', {}).get('session_timeout', 300),
            jpeg_quality=data.get('server', {}).get('jpeg_quality', 75),
            pose_model_path=data.get('pose', {}).get('model_path', 'models/pose_landmarker_full.task'),
            face_model_path=data.get('face', {}).get('model_path', 'models/shape_predictor_68_face_landmarks.dat'),
            camera=data.get('camera', {}),
            pose=data.get('pose', {}),
            skeleton=data.get('skeleton', {}),
            face=data.get('face', {}),
            fall_detector=data.get('fall_detector', {}),
            detection=data.get('detection', {}),
            modes=data.get('modes', [])
        )

    # 🔥 可选：添加 .get() 兼容方法（方便临时修复旧代码）
    def get(self, key: str, default=None):
        """兼容 dict.get() 用法"""
        return getattr(self, key, default)