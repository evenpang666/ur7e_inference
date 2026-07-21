from __future__ import annotations

import logging
from pathlib import Path
import threading
import time
from typing import Callable, Optional

import numpy as np

LOG = logging.getLogger(__name__)


def compose_side_by_side(exterior_rgb: np.ndarray, wrist_rgb: np.ndarray) -> np.ndarray:
    """Return equally sized RGB views concatenated as exterior | wrist."""
    import cv2

    if exterior_rgb.ndim != 3 or wrist_rgb.ndim != 3:
        raise ValueError("Recording frames must be HWC images")
    if exterior_rgb.shape[2] != 3 or wrist_rgb.shape[2] != 3:
        raise ValueError("Recording frames must have three RGB channels")

    output_height = max(exterior_rgb.shape[0], wrist_rgb.shape[0])

    def resize_to_height(frame: np.ndarray) -> np.ndarray:
        if frame.shape[0] == output_height:
            return frame
        width = max(1, round(frame.shape[1] * output_height / frame.shape[0]))
        return cv2.resize(frame, (width, output_height), interpolation=cv2.INTER_LINEAR)

    exterior = resize_to_height(exterior_rgb)
    wrist = resize_to_height(wrist_rgb)
    return np.ascontiguousarray(np.concatenate([exterior, wrist], axis=1), dtype=np.uint8)


class VideoRecorder:
    def __init__(
        self,
        frame_source: Callable[[], tuple[np.ndarray, np.ndarray]],
        output_dir: str | Path,
        fps: float = 30.0,
    ):
        if fps <= 0:
            raise ValueError("Recording FPS must be positive")
        self._frame_source = frame_source
        self._fps = fps
        self._output_dir = Path(output_dir)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="video-recorder", daemon=True)
        self._error: Optional[BaseException] = None
        self._writer = None
        self._frame_count = 0
        self.output_path: Optional[Path] = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start(self) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        candidate = self._output_dir / f"run_{timestamp}.mp4"
        suffix = 1
        while candidate.exists():
            candidate = self._output_dir / f"run_{timestamp}_{suffix}.mp4"
            suffix += 1
        self.output_path = candidate
        self._thread.start()
        LOG.info("Video recording started: %s (%.1f FPS)", candidate.resolve(), self._fps)
        return candidate

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError("Video recording failed") from self._error

    def _run(self) -> None:
        import cv2

        period = 1.0 / self._fps
        next_tick = time.monotonic()
        try:
            while not self._stop.is_set():
                exterior, wrist = self._frame_source()
                frame_rgb = compose_side_by_side(exterior, wrist)
                if self._writer is None:
                    if self.output_path is None:
                        raise RuntimeError("Recording output path is not initialized")
                    height, width = frame_rgb.shape[:2]
                    self._writer = cv2.VideoWriter(
                        str(self.output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self._fps,
                        (width, height),
                    )
                    if not self._writer.isOpened():
                        raise RuntimeError(f"Cannot open video writer for {self.output_path}")
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                self._writer.write(frame_bgr)
                self._frame_count += 1

                next_tick += period
                delay = next_tick - time.monotonic()
                if delay < -period:
                    next_tick = time.monotonic()
                    delay = 0.0
                if delay > 0:
                    self._stop.wait(delay)
        except BaseException as exc:
            self._error = exc
            LOG.exception("Video recording thread stopped with an error")
        finally:
            if self._writer is not None:
                self._writer.release()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=6.0)
        if self._thread.is_alive():
            raise RuntimeError("Video recording thread did not stop")
        self.check()
        if self.output_path is not None:
            LOG.info(
                "Video recording stopped: %s (%d frames)",
                self.output_path.resolve(),
                self._frame_count,
            )
