"""
fisheye.py — Correção de distorção OV5647 130°
Pré-computa os mapas uma vez; remap é O(1) por frame.
"""

import cv2
import numpy as np
from config import FISHEYE, CAMERA_WIDTH, CAMERA_HEIGHT


class FisheyeCorrector:
    def __init__(self, w: int = CAMERA_WIDTH, h: int = CAMERA_HEIGHT):
        cfg = FISHEYE
        K = np.array([
            [cfg["fx"], 0,         cfg["cx"]],
            [0,         cfg["fy"], cfg["cy"]],
            [0,         0,         1        ]
        ], dtype=np.float64)

        D = np.array([[cfg["k1"]], [cfg["k2"]],
                      [cfg["k3"]], [cfg["k4"]]], dtype=np.float64)

        nova_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=0.0)

        self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), nova_K, (w, h), cv2.CV_16SC2)

        self._rotate = cfg.get("rotate_180", False)
        print(f"[FisheyeCorrector] mapas pré-computados {w}x{h} "
              f"fx={cfg['fx']} cy={cfg['cy']} rotate={self._rotate}")

    def correct(self, frame: np.ndarray) -> np.ndarray:
        out = cv2.remap(frame, self._map1, self._map2,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0))
        if self._rotate:
            out = cv2.rotate(out, cv2.ROTATE_180)
        return out
