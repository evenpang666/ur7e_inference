import time

import cv2
import numpy as np

from ur7e_vla.cli import build_parser
from ur7e_vla.config import AppConfig
from ur7e_vla.recording import VideoRecorder, compose_side_by_side


def test_recording_defaults_and_cli_option():
    cfg = AppConfig()
    assert cfg.runtime.record_video is False
    assert cfg.runtime.recording_fps == 30.0
    assert cfg.runtime.recording_dir == "recordings"

    args = build_parser().parse_args(["run", "--record-video", "--recording-dir", "videos"])
    assert args.record_video is True
    assert args.recording_dir == "videos"

    restore_args = build_parser().parse_args(
        ["run", "--restore-initial-state", "--initial-state-episode", "episode_000003"]
    )
    assert restore_args.restore_initial_state is True
    assert restore_args.initial_state_episode == "episode_000003"


def test_compose_side_by_side_resizes_to_a_common_height():
    exterior = np.zeros((4, 6, 3), dtype=np.uint8)
    wrist = np.full((2, 3, 3), 255, dtype=np.uint8)

    combined = compose_side_by_side(exterior, wrist)

    assert combined.shape == (4, 12, 3)
    np.testing.assert_array_equal(combined[:, :6], 0)
    np.testing.assert_array_equal(combined[:, 6:], 255)


def test_video_recorder_writes_frames_and_releases_writer(monkeypatch, tmp_path):
    class FakeWriter:
        def __init__(self):
            self.frames = []
            self.released = False

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames.append(frame.copy())

        def release(self):
            self.released = True

    writer = FakeWriter()
    writer_args = {}

    def make_writer(path, fourcc, fps, size):
        writer_args.update(path=path, fourcc=fourcc, fps=fps, size=size)
        return writer

    monkeypatch.setattr(cv2, "VideoWriter", make_writer)
    monkeypatch.setattr(cv2, "VideoWriter_fourcc", lambda *_: 1234)

    exterior = np.zeros((4, 6, 3), dtype=np.uint8)
    exterior[..., 0] = 10
    wrist = np.zeros((4, 6, 3), dtype=np.uint8)
    wrist[..., 2] = 20
    recorder = VideoRecorder(lambda: (exterior, wrist), tmp_path, fps=30.0)

    output_path = recorder.start()
    deadline = time.monotonic() + 1.0
    while recorder.frame_count < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    recorder.stop()

    assert output_path.parent == tmp_path
    assert output_path.suffix == ".mp4"
    assert writer_args["fps"] == 30.0
    assert writer_args["size"] == (12, 4)
    assert len(writer.frames) >= 2
    assert writer.released is True
    # Recorder inputs are RGB; OpenCV receives BGR.
    assert writer.frames[0][0, 0].tolist() == [0, 0, 10]
