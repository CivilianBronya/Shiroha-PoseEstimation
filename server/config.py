# -*- coding: utf-8 -*-
from dataclasses import dataclass

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    pose_model_path: str = "models/pose_landmarker_full.task"
    session_timeout: float = 300.0  # 秒
    max_concurrent_sessions: int = 10
    # 可扩展：日志路径、模型热加载等