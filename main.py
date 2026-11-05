"""视频、摄像头及图片序列检测、跟踪、可视化和 CSV 输出入口。"""
import argparse
import configparser
import csv
import math
import re
import time
from collections import deque
from pathlib import Path

from config import DetectorConfig, TrackerConfig, VideoConfig


def parse_args():
    defaults = VideoConfig()
    detector = DetectorConfig()
    parser = argparse.ArgumentParser(description="YOLO26 / RF-DETR 篮球检测（可选 Kalman 跟踪）")
    parser.add_argument("--source_type", "--source-type", type=int, choices=(0, 1),
                        default=defaults.source_type, help="0: 视频或摄像头；1: 图片序列目录")
    parser.add_argument("--source", default=defaults.source, help="视频路径、摄像头编号或图片序列目录")
    parser.add_argument("--output", type=Path, default=defaults.output, help="输出 .mp4 路径")
    parser.add_argument("--detector", choices=("yolo26", "rf_detr"), default="rf_detr",
                        help="选择篮球检测器，默认 rf_detr")
    parser.add_argument("--weights", type=Path, default=None,
                        help="权重路径；默认根据 --detector 自动选择")
    parser.add_argument("--rf-short-side", type=int, default=768,
                        help="RF-DETR 短边目标尺寸；--imgsz 仅用于 YOLO26")
    parser.add_argument("--class-id", type=int, default=detector.class_id,
                        help="检测类别编号；默认从权重中自动识别篮球类别")
    parser.add_argument("--device", default=detector.device, help="auto / cpu / cuda:0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-missing", type=int, default=TrackerConfig().max_missing)
    parser.add_argument("--kalman", action="store_true", help="启用 Kalman 跟踪；默认仅运行所选模型检测")
    parser.add_argument("--show", action="store_true", help="显示窗口，按 q 退出")
    args = parser.parse_args()
    if args.weights is None:
        args.weights = (Path(__file__).resolve().parent / "weights/rf-detr-basketball.pth"
                        if args.detector == "rf_detr" else Path(__file__).resolve().parent / "weights/yolo26s_ball_fine_tuning.pt")
    if not 0 < args.conf <= 1 or args.imgsz <= 0 or args.rf_short_side <= 0 or args.max_missing < 0:
        parser.error("conf 应在 (0, 1] 内，imgsz 和 rf-short-side 必须为正，max-missing 不能为负")
    if args.output.suffix.lower() != ".mp4":
        parser.error("output 必须使用 .mp4 后缀")
    if not args.source.isdecimal() and Path(args.source).resolve() == args.output.resolve():
        parser.error("输入和输出路径不能相同")
    return args


def read_image_sequence(source):
    """按文件名自然顺序逐张读取目录内图片；不递归，不跳过损坏帧。"""
    import cv2

    directory = Path(source).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"图片序列目录不存在：{directory}")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def natural_key(path):
        return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                     for part in re.split(r"(\d+)", path.name))

    paths = sorted((p for p in directory.iterdir()
                    if p.is_file() and p.suffix.lower() in extensions),
                   key=lambda p: (natural_key(p), p.name))
    if not paths:
        raise RuntimeError(f"目录中没有支持的图片：{directory}")
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"图片读取失败：{path}")
        yield frame


def image_sequence_fps(source, fallback):
    """读取 SportsMOT img1 同级或目录内的 seqinfo.ini 帧率。"""
    directory = Path(source).expanduser()
    for path in (directory / "seqinfo.ini", directory.parent / "seqinfo.ini"):
        if path.is_file():
            metadata = configparser.ConfigParser()
            try:
                metadata.read_string(path.read_text(encoding="utf-8-sig"))
                fps = metadata.getfloat("Sequence", "frameRate")
                if math.isfinite(fps) and fps > 0:
                    return fps
            except (configparser.Error, ValueError):
                pass
            print(f"无法读取有效帧率：{path}")
    print(f"图片序列使用默认帧率 {fallback} FPS")
    return fallback


def run(args):
    # cuDNN 会动态加载多个子库。先按绝对路径加载同一套虚拟环境库，
    # 避免系统 LD_LIBRARY_PATH 中的旧版 graph 库与新版引擎混用。
    # 必须位于 cv2、torch、ultralytics 导入之前；不修改环境变量或重启进程。
    import ctypes
    import sys
    import sysconfig

    cudnn_libraries = []  # 在整个推理期间保留动态库句柄。
    if sys.platform.startswith("linux") and args.device != "cpu":
        for location in dict.fromkeys(sysconfig.get_path(key) for key in ("purelib", "platlib")):
            library_dir = Path(location) / "nvidia" / "cudnn" / "lib"
            graph = library_dir / "libcudnn_graph.so.9"
            if not graph.is_file():
                continue
            # adv/cnn 会依赖 ops，先加载本环境的 ops，避免系统旧库被抢先加载。
            dependencies = [graph, library_dir / "libcudnn_ops.so.9"]
            libraries = dependencies + sorted(p for p in library_dir.glob("libcudnn*.so.9")
                                              if p not in dependencies)
            for library in libraries:
                try:
                    cudnn_libraries.append(ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL))
                except OSError as exc:
                    raise RuntimeError(f"无法加载虚拟环境 cuDNN 库 {library}：{exc}") from exc
            for handle, library in zip(cudnn_libraries, libraries):
                checks = {"libcudnn_graph.so.9": "cudnnGraphVersionCheck",
                          "libcudnn_ops.so.9": "cudnnOpsVersionCheck",
                          "libcudnn_cnn.so.9": "cudnnCnnVersionCheck",
                          "libcudnn_adv.so.9": "cudnnAdvVersionCheck"}
                if library.name in checks:
                    check = getattr(handle, checks[library.name])
                    check.argtypes, check.restype = [], ctypes.c_int
                    status = check()
                    if status != 0:
                        raise RuntimeError(f"cuDNN 子库版本检查失败：{library.name}，状态码 {status}")
            print(f"已加载当前 Python 环境的 cuDNN：{library_dir}", flush=True)
            break

    import cv2
    if args.detector == "rf_detr":
        from rf_detr_detector import RFDETRDetector as BallDetector
    else:
        from yolo26_detector import YOLO26Detector as BallDetector

    cfg = VideoConfig()
    cap = None
    writer = None
    csv_file = None
    frame_index = 0
    started = time.perf_counter()
    try:
        if args.source_type == 1:
            frames = iter(read_image_sequence(args.source))
            fps = image_sequence_fps(args.source, cfg.fallback_fps)
        else:
            source = int(args.source) if args.source.isdecimal() else args.source
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频源：{args.source}")

            def video_frames():
                while True:
                    ok, image = cap.read()
                    if not ok:
                        return
                    yield image

            frames = iter(video_frames())
            fps = cap.get(cv2.CAP_PROP_FPS)
        frame = next(frames, None)
        if frame is None:
            raise RuntimeError("输入源没有可读取的帧。")
        height, width = frame.shape[:2]
        if not math.isfinite(fps) or fps <= 0:
            fps = cfg.fallback_fps
            print(f"无法读取帧率，使用 {fps} FPS")
        detector = BallDetector(DetectorConfig(weights=args.weights, device=args.device,
                                               confidence=args.conf, image_size=args.imgsz,
                                               class_id=args.class_id, rf_short_side=args.rf_short_side))
        tracker = None
        if args.kalman:
            from kalman_tracker import KalmanBallTracker
            tracker = KalmanBallTracker(TrackerConfig(max_missing=args.max_missing))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"无法创建输出视频：{args.output}")
        csv_path = args.output.with_suffix(".csv")
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        rows = csv.writer(csv_file)
        rows.writerow(["frame", "time_s", "status", "cx", "cy", "offset_x_px",
                       "offset_x_normalized", "missing", "confidence", "inference_ms"])
        trail = deque(maxlen=cfg.trail_length)
        while frame is not None:
            if frame.shape[:2] != (height, width):
                raise RuntimeError("输入帧尺寸发生变化。")
            tick = time.perf_counter()
            detections = detector.detect(frame)
            result = tracker.update(detections, 1.0 / fps) if tracker is not None else None
            inference_ms = (time.perf_counter() - tick) * 1000
            for detection in detections:
                x1, y1, x2, y2 = map(round, detection.bbox)
                color = (120, 120, 120) if tracker is not None else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if tracker is None:
                    cx, cy = detection.center
                    cv2.circle(frame, (round(cx), round(cy)), 3, color, -1)
                    cv2.putText(frame, f"{detector.class_name} {detection.confidence:.2f}",
                                (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    offset = cx - width / 2
                    rows.writerow([frame_index, frame_index / fps, "detected", cx, cy,
                                   offset, offset / (width / 2), "", detection.confidence, inference_ms])
            if tracker is not None:
                cv2.line(frame, (width // 2, 0), (width // 2, height - 1), (255, 180, 0), 1)
            if tracker is None:
                label = f"{args.weights.stem} | detections={len(detections)}"
                if not detections:
                    rows.writerow([frame_index, frame_index / fps, "no_detection",
                                   "", "", "", "", "", "", inference_ms])
            elif result:
                cx, cy = result.center
                offset = cx - width / 2
                normalized = offset / (width / 2)
                color = (0, 255, 0) if result.status == "detected" else (0, 180, 255)
                # 超出画面的预测仍写入 CSV，但不作为有效画内控制目标。
                in_frame = 0 <= cx < width and 0 <= cy < height
                if in_frame:
                    point = (round(cx), round(cy))
                    trail.append(point)
                    cv2.circle(frame, point, 7, color, 2)
                    cv2.line(frame, (width // 2, point[1]), point, color, 2)
                else:
                    trail.clear()
                label = f"{result.status} dx={offset:+.1f}px ({normalized:+.3f}) missing={result.missing}"
                rows.writerow([frame_index, frame_index / fps, result.status, cx, cy,
                               offset, normalized, result.missing,
                               result.confidence if result.confidence is not None else "", inference_ms])
            else:
                trail.clear()
                label = "lost / searching"
                rows.writerow([frame_index, frame_index / fps, "lost", "", "", "", "", "", "", inference_ms])
            for p1, p2 in zip(trail, list(trail)[1:]):
                cv2.line(frame, p1, p2, (255, 200, 0), 2)
            cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"frame={frame_index} processing={1000 / max(inference_ms, .001):.1f} FPS",
                        (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            writer.write(frame)
            frame_index += 1
            if args.show:
                cv2.imshow("Basketball tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame = next(frames, None)
        print(f"完成：{frame_index} 帧，耗时 {time.perf_counter()-started:.1f}s")
        print(f"视频：{args.output.resolve()}\n数据：{csv_path.resolve()}")
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        run(arguments)
    except (RuntimeError, ValueError, FileNotFoundError, ImportError) as exc:
        raise SystemExit(f"运行失败：{exc}") from exc
