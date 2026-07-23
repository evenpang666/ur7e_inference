from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np

from .config import CameraConfig


class LatestFrame:
    def __init__(self, read_frame: Callable[[], np.ndarray], name: str, max_age_s: float):
        self._read_frame = read_frame
        self._name = name
        self._frame: Optional[np.ndarray] = None
        self._frame_time: Optional[float] = None
        self._error: Optional[Exception] = None
        self._max_age_s = max_age_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"camera-{name}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._read_frame()
                if frame is None or frame.size == 0:
                    raise RuntimeError(f"{self._name} returned an empty frame")
                with self._lock:
                    self._frame = frame
                    self._frame_time = time.monotonic()
                    self._error = None
            except Exception as exc:
                with self._lock:
                    self._error = exc
                time.sleep(0.05)

    def get(self, timeout_s: float = 5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                is_fresh = self._frame_time is not None and time.monotonic() - self._frame_time <= self._max_age_s
                if self._frame is not None and is_fresh:
                    return self._frame.copy()
                error = self._error
            if error is not None and time.monotonic() + 0.1 >= deadline:
                raise RuntimeError(f"Failed to read {self._name}") from error
            time.sleep(0.01)
        raise TimeoutError(f"Timed out waiting for {self._name}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


class CameraPair:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._pipeline = None
        self._wrist_pipeline = None
        self._capture = None
        self._exterior: Optional[LatestFrame] = None
        self._wrist: Optional[LatestFrame] = None

    def start(self) -> None:
        if self._exterior is not None or self._wrist is not None:
            return
        import cv2
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        rs_cfg = rs.config()
        if self.cfg.realsense_serial:
            rs_cfg.enable_device(self.cfg.realsense_serial)
        rs_cfg.enable_stream(
            rs.stream.color,
            self.cfg.realsense_width,
            self.cfg.realsense_height,
            rs.format.bgr8,
            self.cfg.realsense_fps,
        )
        pipeline.start(rs_cfg)
        self._pipeline = pipeline

        wrist_pipeline = None
        capture = None
        if self.cfg.wrist_realsense_serial:
            wrist_pipeline = rs.pipeline()
            wrist_rs_cfg = rs.config()
            wrist_rs_cfg.enable_device(self.cfg.wrist_realsense_serial)
            wrist_rs_cfg.enable_stream(
                rs.stream.color,
                self.cfg.wrist_width,
                self.cfg.wrist_height,
                rs.format.bgr8,
                self.cfg.wrist_fps,
            )
            try:
                wrist_pipeline.start(wrist_rs_cfg)
            except Exception:
                pipeline.stop()
                raise
            self._wrist_pipeline = wrist_pipeline
        else:
            device = self.cfg.wrist_device
            if isinstance(device, str) and device.isdigit():
                device = int(device)
            capture = cv2.VideoCapture(device)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.wrist_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.wrist_height)
            capture.set(cv2.CAP_PROP_FPS, self.cfg.wrist_fps)
            if not capture.isOpened():
                pipeline.stop()
                raise RuntimeError(f"Cannot open Pika wrist camera {device!r}")
            self._capture = capture

        def read_exterior() -> np.ndarray:
            frames = pipeline.wait_for_frames(2000)
            color = frames.get_color_frame()
            if not color:
                raise RuntimeError("D435i color frame missing")
            return cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_BGR2RGB)

        def read_wrist() -> np.ndarray:
            if wrist_pipeline is not None:
                frames = wrist_pipeline.wait_for_frames(2000)
                color = frames.get_color_frame()
                if not color:
                    raise RuntimeError("Pika wrist RealSense color frame missing")
                return cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_BGR2RGB)
            assert capture is not None
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Pika wrist camera read failed")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self._exterior = LatestFrame(read_exterior, "D435i", self.cfg.max_frame_age_s)
        self._wrist = LatestFrame(read_wrist, "Pika wrist camera", self.cfg.max_frame_age_s)
        self._exterior.start()
        self._wrist.start()

    def frames(self) -> tuple[np.ndarray, np.ndarray]:
        if self._exterior is None or self._wrist is None:
            raise RuntimeError("Cameras have not been started")
        return self._exterior.get(), self._wrist.get()

    def stop(self) -> None:
        if self._exterior:
            self._exterior.stop()
        if self._wrist:
            self._wrist.stop()
        if self._capture:
            self._capture.release()
        if self._pipeline:
            self._pipeline.stop()
        if self._wrist_pipeline:
            self._wrist_pipeline.stop()
        self._pipeline = None
        self._wrist_pipeline = None
        self._capture = None
        self._exterior = None
        self._wrist = None


def resize_with_pad(image: np.ndarray, size: int) -> np.ndarray:
    import cv2

    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x = (size - new_width) // 2
    y = (size - new_height) // 2
    canvas[y : y + new_height, x : x + new_width] = resized.astype(np.uint8)
    return canvas


def list_opencv_cameras(max_index: int = 10) -> list[int]:
    import cv2

    found = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            ok, _ = capture.read()
            if ok:
                found.append(index)
        capture.release()
    return found
