"""基于 PyTorch / YOLO26 的篮球检测，兼容已有检测权重。"""
from pathlib import Path
import shutil

import numpy as np

from config import Detection, DetectorConfig


class YOLO26Detector:
    def __init__(self, config: DetectorConfig):
        import torch
        from ultralytics import YOLO

        self.config = config
        self.device = config.device
        if self.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        weights = Path(config.weights).expanduser().resolve()
        weights.parent.mkdir(parents=True, exist_ok=True)
        if not weights.is_file():
            official_weights = {f"yolo26{scale}.pt" for scale in "nsmlx"}
            if weights.name not in official_weights:
                raise FileNotFoundError(f"权重文件不存在：{weights}")
            # 官方下载器直接保存到 weights 目录，避免污染当前工作目录。
            from ultralytics.utils.downloads import attempt_download_asset
            print(f"首次使用，下载官方预训练权重：{weights}", flush=True)
            downloaded = Path(attempt_download_asset(str(weights))).resolve()
            if not downloaded.is_file():
                raise FileNotFoundError(f"权重下载未完成，请检查网络或手动下载到：{weights}")
            if downloaded != weights:
                shutil.copy2(downloaded, weights)
        self.model = YOLO(str(weights))
        if self.model.task != "detect":
            raise ValueError("请使用 YOLO 目标检测权重。")
        self.class_id = config.class_id
        if self.class_id is None:
            candidates = [index for index, name in self.model.names.items()
                          if name.strip().lower() in {"ball", "basketball", "sports ball"}]
            if len(candidates) != 1:
                raise ValueError(f"无法唯一识别篮球类别：{self.model.names}，请指定 --class-id。")
            self.class_id = candidates[0]
        if self.class_id not in self.model.names:
            raise ValueError(f"类别编号不存在：{self.class_id}，当前类别：{self.model.names}")
        self.class_name = self.model.names[self.class_id]
        print(f"模型：{weights.name}，检测设备：{self.device}，类别：{self.class_id}: {self.class_name}", flush=True)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            source=frame, classes=[self.class_id],
            conf=self.config.confidence, iou=self.config.iou,
            imgsz=self.config.image_size, device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        return [Detection(tuple(map(float, box)), float(score))
                for box, score in zip(boxes, scores)]
