"""项目默认参数；命令行参数可覆盖常用选项。"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class DetectorConfig:
    weights: Path = ROOT / "weights" / "rf-detr-basketball.pth"
    rf_short_side: int = 768
    confidence: float = 0.25
    iou: float = 0.45
    image_size: int = 1280
    device: str = "auto"
    class_id: int | None = None  # 自动识别 ball / basketball / COCO sports ball


@dataclass
class TrackerConfig:
    max_missing: int = 15
    gate_distance: float = 150.0  # 原始视频像素
    mahalanobis_threshold: float = 9.21  # 二维观测约 99% 门限
    acceleration_std: float = 600.0  # px/s²
    measurement_std: float = 10.0  # px
    initial_velocity_std: float = 500.0  # px/s


@dataclass
class VideoConfig:
    source_type: int = 0  # 0: 视频或摄像头；1: 图片序列目录
    source: str = str(ROOT / "dataset" / "basketball.mp4")
    output: Path = ROOT / "outputs" / "tracked.mp4"
    fallback_fps: float = 30.0
    trail_length: int = 40


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2
