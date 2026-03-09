# -*- coding: utf-8 -*-
import base64
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def encode_frame_to_base64(frame_bgr: np.ndarray, quality: int = 75) -> str:
    """
    将 OpenCV BGR 帧编码为 base64 JPEG 字符串
    :param frame_bgr: numpy array (H,W,3) BGR format
    :param quality: JPEG 质量 1-100
    :return: base64 字符串，失败返回 None
    """
    try:
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode('ascii')
    except Exception as e:
        logger.error(f"Frame encode error: {e}")
        return None

def decode_base64_to_frame(b64_str: str) -> np.ndarray:
    """
    将 base64 JPEG 字符串解码为 OpenCV BGR 帧
    :param b64_str: base64 字符串
    :return: numpy array 或 None
    """
    try:
        img_data = base64.b64decode(b64_str)
        frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Frame decode error: {e}")
        return None