"""独立推理线程：模型加载、检测、跟踪、标注和结果输出。"""
import csv
import logging
import time
from pathlib import Path
from collections import deque

from config import DetectorConfig, TrackerConfig, VideoConfig


def prepare_runtime(args):
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

    return cudnn_libraries


def inference_worker(args, state):
    writer = None
    csv_file = None
    try:
        with state.condition:
            state.condition.wait_for(lambda: state.latest is not None or state.stop.is_set())
            if state.stop.is_set():
                return
            origin = state.first_received_at if state.first_received_at is not None else state.latest.captured_at
        import cv2
        if args.detector == "rf_detr":
            from rf_detr_detector import RFDETRDetector as BallDetector
        else:
            from yolo26_detector import YOLO26Detector as BallDetector
        detector = BallDetector(DetectorConfig(weights=args.weights, device=args.device,
                                               confidence=args.conf, image_size=args.imgsz,
                                               class_id=args.class_id, rf_short_side=args.rf_short_side))
        tracker = None
        if args.kalman:
            from kalman_tracker import KalmanBallTracker
            tracker = KalmanBallTracker(TrackerConfig(max_missing=args.max_missing))
        cfg = VideoConfig()
        trail = deque(maxlen=cfg.trail_length)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8")
        rows = csv.writer(csv_file)
        rows.writerow(["frame", "time_s", "status", "cx", "cy", "offset_x_px",
                       "offset_x_normalized", "missing", "confidence", "inference_ms",
                       "inference_start_s", "inference_end_s"])
        base_rows = rows
        class TimedRows:
            def writerow(self, values):
                base_rows.writerow(values + [inference_start - origin, inference_end - origin])
        rows = TimedRows()
        previous = None
        output_size = None
        while not state.stop.is_set():
            with state.condition:
                state.condition.wait_for(lambda: state.pending is not None or state.stop.is_set())
                if state.stop.is_set():
                    break
                packet = state.pending
                state.pending = None
            frame = packet.image.copy()
            height, width = frame.shape[:2]
            if previous is not None and (previous.generation != packet.generation
                                        or previous.image.shape != packet.image.shape):
                if tracker is not None:
                    tracker.reset()
                trail.clear()
                previous = None
            inference_start = time.monotonic()
            detections = detector.detect(frame)
            inference_end = time.monotonic()
            inference_ms = (inference_end - inference_start) * 1000
            logging.info("预测延迟 %.1f ms", inference_ms)
            dt = packet.captured_at - previous.captured_at if previous is not None else 1 / 30
            result = tracker.update(detections, max(dt, 1e-6)) if tracker is not None else None
            previous = packet
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
                    rows.writerow([packet.index, packet.captured_at - origin, "detected", cx, cy,
                                   offset, offset / (width / 2), "", detection.confidence, inference_ms])
            if tracker is not None:
                cv2.line(frame, (width // 2, 0), (width // 2, height - 1), (255, 180, 0), 1)
            if tracker is None:
                label = f"{args.weights.stem} | detections={len(detections)}"
                if not detections:
                    rows.writerow([packet.index, packet.captured_at - origin, "no_detection",
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
                rows.writerow([packet.index, packet.captured_at - origin, result.status, cx, cy,
                               offset, normalized, result.missing,
                               result.confidence if result.confidence is not None else "", inference_ms])
            else:
                trail.clear()
                label = "lost / searching"
                rows.writerow([packet.index, packet.captured_at - origin, "lost", "", "", "", "", "", "", inference_ms])
            for p1, p2 in zip(trail, list(trail)[1:]):
                cv2.line(frame, p1, p2, (255, 200, 0), 2)
            cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Inference: {inference_ms:.1f} ms", (15, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            with state.condition:
                if state.connected_since is not None and state.latest.generation == packet.generation:
                    state.prediction = (packet, frame)
            if writer is None:
                output_size = (width, height)
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                         cfg.fallback_fps, output_size)
                if not writer.isOpened():
                    raise RuntimeError("无法创建输出视频")
            saved = frame if (width, height) == output_size else cv2.resize(frame, output_size)
            writer.write(saved)
            csv_file.flush()
    except Exception as exc:
        state.fail(exc)
    finally:
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
