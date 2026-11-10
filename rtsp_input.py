"""独立 RTSP 采集；仅保留最新输入帧和最新待预测帧。"""
import logging
import threading
import time
from dataclasses import dataclass

def open_stream(cv2, url, timeout):
    """返回 VideoCapture；调用方负责 release。超时参数必须在打开时设置。"""
    timeout_ms = max(1, int(timeout * 1000))
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms,
    ])


@dataclass(frozen=True)
class FramePacket:
    index: int
    captured_at: float
    generation: int
    image: object


class StreamState:
    def __init__(self):
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.latest = None
        self.pending = None
        self.prediction = None
        self.error = None
        self.connected_since = None
        self.connected_seconds = 0.0
        self.first_received_at = None

    def fail(self, exc):
        with self.condition:
            self.error = exc
            self.stop.set()
            self.condition.notify_all()

    def duration(self):
        with self.condition:
            return self.connected_seconds + (
                time.monotonic() - self.connected_since if self.connected_since is not None else 0)


def capture_stream(cv2, url, step, state):
    cap = None
    index = -1
    failures = 0
    generation = 0
    try:
        while not state.stop.is_set():
            if cap is None:
                logging.info("正在连接手机 RTSP...")
                cap = open_stream(cv2, url, 5.0)
            ok, image = cap.read() if cap.isOpened() else (False, None)
            now = time.monotonic()
            if state.stop.is_set():
                break
            if not ok or image is None or image.size == 0:
                cap.release()
                cap = None
                with state.condition:
                    if state.connected_since is not None:
                        state.connected_seconds += now - state.connected_since
                        state.connected_since = None
                    state.pending = None
                    state.prediction = None
                failures += 1
                if failures > 3:
                    raise RuntimeError("RTSP 连接/读取失败，默认 3 次重试已用尽")
                logging.warning("RTSP 中断，准备重试 %d/3", failures)
                state.stop.wait(1.0)
                continue
            failures = 0
            index += 1
            with state.condition:
                if state.connected_since is None:
                    generation += 1
                    state.connected_since = now
                    logging.info("RTSP 已连接")
                packet = FramePacket(index, now, generation, image)
                if state.first_received_at is None:
                    state.first_received_at = now
                state.latest = packet
                if step == 0 or (index + 1) % step == 0:
                    state.pending = packet
                state.condition.notify_all()
    except Exception as exc:
        state.fail(exc)
    finally:
        if cap is not None:
            cap.release()
        with state.condition:
            if state.connected_since is not None:
                state.connected_seconds += time.monotonic() - state.connected_since
                state.connected_since = None
            state.condition.notify_all()
