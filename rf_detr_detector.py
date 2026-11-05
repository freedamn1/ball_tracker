"""RF-DETR 篮球定位，适配现有 detect(BGR) -> list[Detection] 接口。"""
from pathlib import Path

import numpy as np

from config import Detection, DetectorConfig


class RFDETRDetector:
    def __init__(self, config: DetectorConfig):
        import torch
        from rfdetr import from_checkpoint

        self.config = config
        self.device = config.device
        if self.device == 'auto':
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        weights = Path(config.weights).expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f'RF-DETR 权重不存在：{weights}')
        with weights.open('rb') as stream:
            if stream.read(100).startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise ValueError(f'RF-DETR 权重是 Git LFS 指针，请下载真实 .pth 文件：{weights}')
        self.model = from_checkpoint(str(weights), device=self.device)
        names = getattr(self.model, 'class_names', {})
        entries = names.items() if isinstance(names, dict) else enumerate(names)
        names = {int(key): str(name) for key, name in entries}
        self.class_id = config.class_id
        if self.class_id is None:
            candidates = [key for key, name in names.items()
                          if name.strip().lower().replace('_', ' ').replace('-', ' ') == 'ball']
            if len(candidates) != 1:
                raise ValueError(f'无法唯一识别 RF-DETR ball 类别：{names}，请指定 --class-id')
            self.class_id = candidates[0]
        if self.class_id not in names:
            raise ValueError(f'类别编号不存在：{self.class_id}，当前类别：{names}')
        self.class_name = names[self.class_id]
        print(f'模型：{weights.name}，检测设备：{self.device}，'
              f'类别：{self.class_id}: {self.class_name}', flush=True)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        import cv2
        import torch

        height, width = frame.shape[:2]
        scale = min(1.0, self.config.rf_short_side / min(height, width), 1333 / max(height, width))
        shape = (max(32, int(height * scale) // 32 * 32),
                 max(32, int(width * scale) // 32 * 32))
        with torch.inference_mode():
            result = self.model.predict(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                threshold=self.config.confidence, shape=shape, include_source_image=False,
            )
        if result is None or result.class_id is None or len(result) == 0:
            return []
        return [Detection(tuple(map(float, box)), float(score))
                for box, class_id, score in zip(result.xyxy, result.class_id, result.confidence)
                if int(class_id) == self.class_id and float(score) >= self.config.confidence]
