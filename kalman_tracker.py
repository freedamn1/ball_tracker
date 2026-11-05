"""单篮球跟踪：匀速 Kalman、候选关联、预测和丢失管理。"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import Detection, TrackerConfig


@dataclass(frozen=True)
class TrackResult:
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    status: str  # detected / predicted
    missing: int
    confidence: Optional[float]


class KalmanBallTracker:
    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        if (self.config.max_missing < 0 or self.config.gate_distance <= 0
                or self.config.mahalanobis_threshold <= 0
                or self.config.measurement_std <= 0
                or self.config.acceleration_std < 0
                or self.config.initial_velocity_std <= 0):
            raise ValueError("跟踪参数必须为有效的正值，max_missing 和加速度可为零。")
        self.H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        self.R = np.eye(2) * self.config.measurement_std ** 2
        self.reset()

    def reset(self):
        self.x = None
        self.P = None
        self.size = None
        self.missing = 0

    def _initialize(self, detection: Detection):
        self.x = np.array([*detection.center, 0., 0.], dtype=float)
        self.P = np.diag([self.config.measurement_std ** 2] * 2
                         + [self.config.initial_velocity_std ** 2] * 2)
        x1, y1, x2, y2 = detection.bbox
        self.size = (x2 - x1, y2 - y1)
        self.missing = 0

    def update(self, detections: list[Detection], dt: float) -> Optional[TrackResult]:
        """每个视频帧调用一次，dt 使用视频时间（秒），而非推理耗时。"""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt 必须是有限的正数。")
        selected = None
        if self.x is None:
            if not detections:
                return None
            selected = max(detections, key=lambda d: d.confidence)
            self._initialize(selected)
        else:
            F = np.eye(4)
            F[0, 2] = F[1, 3] = dt
            G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2],
                          [dt, 0], [0, dt]])
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + G @ G.T * self.config.acceleration_std ** 2
            S = self.H @ self.P @ self.H.T + self.R
            candidates = []
            for detection in detections:
                residual = np.asarray(detection.center) - self.x[:2]
                distance = float(residual @ np.linalg.solve(S, residual))
                if (np.linalg.norm(residual) <= self.config.gate_distance
                        and distance <= self.config.mahalanobis_threshold):
                    candidates.append((distance, -detection.confidence, detection))
            if candidates:
                selected = min(candidates, key=lambda c: c[:2])[2]
                K = np.linalg.solve(S, self.H @ self.P).T
                self.x += K @ (np.asarray(selected.center) - self.H @ self.x)
                # Joseph 形式保持协方差对称和数值稳定。
                A = np.eye(4) - K @ self.H
                self.P = A @ self.P @ A.T + K @ self.R @ K.T
                x1, y1, x2, y2 = selected.bbox
                self.size = (x2 - x1, y2 - y1)
                self.missing = 0
            else:
                self.missing += 1
                if self.missing > self.config.max_missing:
                    self.reset()
                    return None
        cx, cy = map(float, self.x[:2])
        w, h = self.size
        return TrackResult((cx, cy), (cx-w/2, cy-h/2, cx+w/2, cy+h/2),
                           "detected" if selected else "predicted", self.missing,
                           selected.confidence if selected else None)
