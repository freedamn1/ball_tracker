"""RTSP 实时采集、独立推理与双窗口预览入口。"""
import argparse
import logging
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from config import DetectorConfig, TrackerConfig, VideoConfig
from rtsp_input import StreamState, capture_stream
from inference_worker import inference_worker, prepare_runtime


def parse_args():
    defaults = VideoConfig()
    detector = DetectorConfig()
    parser = argparse.ArgumentParser(description="YOLO26 / RF-DETR 篮球检测（可选 Kalman 跟踪）")
    parser.add_argument("--rtsp-url", default=os.getenv("RTSP_URL"), help="手机 RTSP 完整地址")
    parser.add_argument("--frame-step", type=int, default=0, help="0: 不跳帧；5: 每 5 帧选取一次")
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
    args = parser.parse_args()
    if args.weights is None:
        args.weights = (Path(__file__).resolve().parent / "weights/rf-detr-basketball.pth"
                        if args.detector == "rf_detr" else Path(__file__).resolve().parent / "weights/yolo26s_ball_fine_tuning.pt")
    if not 0 < args.conf <= 1 or args.imgsz <= 0 or args.rf_short_side <= 0 or args.max_missing < 0:
        parser.error("conf 应在 (0, 1] 内，imgsz 和 rf-short-side 必须为正，max-missing 不能为负")
    if args.output.suffix.lower() != ".mp4":
        parser.error("output 必须使用 .mp4 后缀")
    if args.frame_step < 0:
        parser.error("frame-step 不能为负")
    try:
        parts = urlsplit(args.rtsp_url or "")
        valid = parts.scheme.lower() == "rtsp" and parts.hostname and parts.port != 0
    except ValueError:
        valid = False
    if not valid:
        parser.error("请指定有效的 --rtsp-url，或设置 RTSP_URL 环境变量")
    return args



def run(args):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    handles = prepare_runtime(args)
    import cv2
    import numpy as np

    state = StreamState()
    threads = [threading.Thread(target=capture_stream, args=(cv2, args.rtsp_url, args.frame_step, state),
                                name="rtsp-capture"),
               threading.Thread(target=inference_worker, args=(args, state), name="inference")]
    windows = ("RTSP input", "Detection result")
    placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Connecting / loading model...", (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    started_threads = []
    try:
        for window in windows:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.imshow(window, placeholder)
        for thread in threads:
            thread.start()
            started_threads.append(thread)
        next_report = time.monotonic() + 10
        shown_input = None
        shown_result = None
        while not state.stop.is_set():
            with state.condition:
                latest, prediction = state.latest, state.prediction
            if latest is not None and latest is not shown_input:
                cv2.imshow(windows[0], latest.image)
                shown_input = latest
            if prediction is not None and prediction is not shown_result:
                cv2.imshow(windows[1], prediction[1])
                shown_result = prediction
            elif prediction is None and shown_result is not None:
                cv2.imshow(windows[1], placeholder)
                shown_result = None
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if any(cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 for window in windows):
                break
            if time.monotonic() >= next_report:
                logging.info("累计 RTSP 连接时长 %.1f 秒", state.duration())
                next_report = time.monotonic() + 10
    finally:
        state.stop.set()
        with state.condition:
            state.condition.notify_all()
        cv2.destroyAllWindows()
        for thread in started_threads:
            thread.join()
        logging.info("累计 RTSP 连接时长 %.1f 秒", state.duration())
    if state.error is not None:
        raise RuntimeError(f"后台任务失败：{type(state.error).__name__}，请检查连接、模型和输出配置") from state.error


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(parse_args())
    except KeyboardInterrupt:
        pass
