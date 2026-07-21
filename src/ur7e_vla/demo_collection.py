from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import re
import shutil
import threading
import time
import uuid
from typing import Callable, Optional

import numpy as np

from .cameras import CameraPair
from .config import AppConfig, DemoConfig
from .hardware import PikaGripper, UR7e

LOG = logging.getLogger(__name__)


def normalize_task(task: str) -> str:
    return "_".join(task.strip().lower().replace("-", " ").split())


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3) + _skew(vector)
    axis = vector / angle
    cross = _skew(axis)
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def matrix_to_rotation_vector(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    cos_angle = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return np.asarray(
            [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
        ) / 2.0
    if abs(math.pi - angle) < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eigh((matrix + np.eye(3)) / 2.0)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        return axis * angle
    axis = np.asarray(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
    ) / (2.0 * math.sin(angle))
    return axis * angle


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"Expected xyz+rotation-vector pose, got {pose.shape}")
    result = np.eye(4)
    result[:3, :3] = rotation_vector_to_matrix(pose[3:])
    result[:3, 3] = pose[:3]
    return result


def matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate([matrix[:3, 3], matrix_to_rotation_vector(matrix[:3, :3])])


@dataclass(frozen=True)
class SensorSample:
    monotonic_s: float
    tracker_pose: np.ndarray
    gripper: float


@dataclass(frozen=True)
class DemonstrationTrajectory:
    samples: tuple[SensorSample, ...]
    fps: int

    def __post_init__(self) -> None:
        if len(self.samples) < 2:
            raise ValueError("A demonstration trajectory needs at least two samples")


class PikaSenseSource:
    """Lazy Pika Sense wrapper so a missing Lighthouse backend has a useful error."""

    def __init__(self, cfg: DemoConfig):
        self.cfg = cfg
        self._sense = None
        self._device: Optional[str] = cfg.tracker_device
        self._last_tracker_timestamp = None
        self._last_update_monotonic = 0.0

    @property
    def is_ready(self) -> bool:
        return bool(self._sense is not None and self._sense.is_connected and self._device is not None)

    @property
    def tracker_device(self) -> Optional[str]:
        return self._device

    @staticmethod
    def _is_lighthouse(name: str) -> bool:
        # libsurvive exposes base stations as LH0/LH1. They have a pose too,
        # but are fixed infrastructure rather than the hand-held Pika tracker.
        return re.fullmatch(r"lh\d+", name.strip(), flags=re.IGNORECASE) is not None

    def connect(self) -> None:
        # A previous OOTX timeout can leave the SDK reader threads alive. Tear
        # that attempt down before reconnecting so retries are genuine retries.
        self.close()
        self._device = self.cfg.tracker_device
        self._last_tracker_timestamp = None
        self._last_update_monotonic = 0.0
        try:
            from pika.sense import Sense
        except ImportError as exc:
            raise RuntimeError("Pika SDK missing; install agx-pypika") from exc
        try:
            import pysurvive  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Pika Lighthouse tracking requires pysurvive, but it is not available in this Python environment. "
                "Use a supported Linux environment with libsurvive/pysurvive and USB access to both base stations."
            ) from exc
        try:
            self._sense = Sense(self.cfg.pika_sense_port)
            if not self._sense.connect():
                raise RuntimeError(f"Cannot connect to Pika Sense at {self.cfg.pika_sense_port}")
            tracker = self._sense.get_vive_tracker()
            if tracker is None or not tracker.running:
                raise RuntimeError("Pika Vive Tracker could not start")
            deadline = time.monotonic() + self.cfg.tracker_start_timeout_s
            last_pose_names: list[str] = []
            last_device_names: list[str] = []
            last_device_info: dict = {}
            while time.monotonic() < deadline:
                poses = tracker.get_pose()
                last_pose_names = sorted(poses)
                # get_pose() only exposes devices after their first valid pose.
                # Keep the raw device list as a diagnostic, but never select a
                # raw device until it has a valid pose.
                try:
                    last_device_names = sorted(tracker.get_devices())
                    last_device_info = tracker.get_device_info() or {}
                except (AttributeError, TypeError):
                    pass
                if self._device is not None:
                    if self._device in poses:
                        LOG.info("Using configured Pika tracker device %s", self._device)
                        return
                else:
                    candidates = [name for name in last_pose_names if not self._is_lighthouse(name)]
                    if len(candidates) == 1:
                        self._device = candidates[0]
                        LOG.info("Using detected Pika tracker device %s", self._device)
                        return
                    if len(candidates) > 1:
                        raise RuntimeError(
                            "Multiple non-Lighthouse tracked devices found: "
                            f"{candidates}. Set demo.tracker_device explicitly."
                        )
                time.sleep(0.1)
            raise TimeoutError(
                "No hand-held Pika tracker pose received before timeout. "
                f"Valid-pose devices: {last_pose_names or 'none'}; "
                f"detected devices: {last_device_names or 'none'}; "
                f"device stats: {last_device_info or 'none'}. "
                "A detected device with zero updates means Lighthouse geometry has not solved; "
                "LH0/LH1 are base stations, not the sensor."
            )
        except BaseException:
            self.close()
            raise

    def sample(self) -> SensorSample:
        if self._sense is None or self._device is None:
            raise RuntimeError("Pika Sense is not connected")
        pose = self._sense.get_pose(self._device)
        if pose is None:
            raise RuntimeError(f"Tracker {self._device!r} has no pose")
        now = time.monotonic()
        if pose.timestamp != self._last_tracker_timestamp:
            self._last_tracker_timestamp = pose.timestamp
            self._last_update_monotonic = now
        if now - self._last_update_monotonic > self.cfg.max_tracker_age_s:
            raise TimeoutError(f"Tracker pose is stale for {now - self._last_update_monotonic:.3f}s")
        distance = float(self._sense.get_gripper_distance())
        span = self.cfg.gripper_distance_max_mm - self.cfg.gripper_distance_min_mm
        gripper = float(np.clip((distance - self.cfg.gripper_distance_min_mm) / span, 0.0, 1.0))
        if self.cfg.gripper_invert:
            gripper = 1.0 - gripper
        return SensorSample(
            monotonic_s=now,
            tracker_pose=np.asarray([*pose.position, *_quaternion_to_rotation_vector(pose.rotation)], dtype=np.float64),
            gripper=gripper,
        )

    def close(self) -> None:
        if self._sense is not None:
            self._sense.disconnect()
            self._sense = None
        self._device = self.cfg.tracker_device
        self._last_tracker_timestamp = None
        self._last_update_monotonic = 0.0


def _quaternion_to_rotation_vector(quaternion_xyzw) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or norm < 1e-12:
        raise ValueError("Invalid tracker quaternion")
    quaternion /= norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[:3]))
    if vector_norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(vector_norm, float(quaternion[3]))
    return quaternion[:3] / vector_norm * angle


def record_sensor_trajectory(
    source: PikaSenseSource,
    fps: int,
    stop_event: threading.Event,
    on_sample: Optional[Callable[[int], None]] = None,
) -> DemonstrationTrajectory:
    period = 1.0 / fps
    next_tick = time.monotonic()
    samples: list[SensorSample] = []
    while not stop_event.is_set():
        samples.append(source.sample())
        if on_sample is not None:
            on_sample(len(samples))
        next_tick += period
        stop_event.wait(max(0.0, next_tick - time.monotonic()))
    return DemonstrationTrajectory(tuple(samples), fps)


def map_tracker_trajectory(
    trajectory: DemonstrationTrajectory,
    robot_anchor_pose: np.ndarray,
    cfg: DemoConfig,
) -> list[np.ndarray]:
    tracker_anchor = pose_to_matrix(trajectory.samples[0].tracker_pose)
    robot_anchor = pose_to_matrix(robot_anchor_pose)
    targets = []
    for sample in trajectory.samples:
        delta = np.linalg.inv(tracker_anchor) @ pose_to_matrix(sample.tracker_pose)
        translation = delta[:3, 3] * cfg.translation_scale
        rotation_vector = matrix_to_rotation_vector(delta[:3, :3]) * cfg.rotation_scale
        if np.linalg.norm(translation) > cfg.max_translation_m:
            raise RuntimeError("Demonstration exceeds configured translation workspace limit")
        if np.linalg.norm(rotation_vector) > cfg.max_rotation_rad:
            raise RuntimeError("Demonstration exceeds configured rotation workspace limit")
        scaled_delta = np.eye(4)
        scaled_delta[:3, :3] = rotation_vector_to_matrix(rotation_vector)
        scaled_delta[:3, 3] = translation
        targets.append(matrix_to_pose(robot_anchor @ scaled_delta))
    return targets


def calibration_file(cfg: DemoConfig) -> Path:
    return Path(cfg.calibration_path).expanduser().resolve()


def _average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    mean = sum(rotations) / len(rotations)
    left, _, right = np.linalg.svd(mean)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def _hand_eye_residual(
    sensor_poses: list[np.ndarray], tcp_poses: list[np.ndarray], sensor_to_tcp: np.ndarray
) -> np.ndarray:
    """Residual for inv(T_tcp0) T_tcpi = inv(B) inv(T_sensor0) T_sensori B."""
    sensor_zero = pose_to_matrix(sensor_poses[0])
    tcp_zero = pose_to_matrix(tcp_poses[0])
    residuals: list[np.ndarray] = []
    inverse_extrinsic = np.linalg.inv(sensor_to_tcp)
    for sensor_pose, tcp_pose in zip(sensor_poses[1:], tcp_poses[1:], strict=True):
        sensor_delta = np.linalg.inv(sensor_zero) @ pose_to_matrix(sensor_pose)
        tcp_delta = np.linalg.inv(tcp_zero) @ pose_to_matrix(tcp_pose)
        error = np.linalg.inv(tcp_delta) @ inverse_extrinsic @ sensor_delta @ sensor_to_tcp
        pose_error = matrix_to_pose(error)
        # Balance metres and radians while preserving an interpretable residual.
        residuals.append(np.concatenate([pose_error[:3] / 0.05, pose_error[3:] / 0.25]))
    return np.concatenate(residuals)


def solve_sensor_to_tcp_hand_eye(
    sensor_poses: list[np.ndarray], tcp_poses: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Solve ``T_tcp = A @ T_sensor @ B`` from at least three paired poses.

    ``A`` is Lighthouse-to-UR-base and ``B`` is the fixed Sensor-to-TCP
    extrinsic. A small NumPy Gauss-Newton solve avoids requiring SciPy in the
    real-time collector environment.
    """
    if len(sensor_poses) != len(tcp_poses) or len(sensor_poses) < 3:
        raise ValueError("Hand-eye calibration needs at least three Sensor/TCP pose pairs")
    if any(np.asarray(pose).shape != (6,) for pose in [*sensor_poses, *tcp_poses]):
        raise ValueError("Hand-eye calibration poses must all be six-dimensional")

    parameters = np.zeros(6, dtype=np.float64)  # pose representation of B
    epsilon = 1e-5
    for _ in range(40):
        extrinsic = pose_to_matrix(parameters)
        residual = _hand_eye_residual(sensor_poses, tcp_poses, extrinsic)
        jacobian = np.empty((residual.size, 6), dtype=np.float64)
        for axis in range(6):
            shifted = parameters.copy()
            shifted[axis] += epsilon
            jacobian[:, axis] = (
                _hand_eye_residual(sensor_poses, tcp_poses, pose_to_matrix(shifted)) - residual
            ) / epsilon
        step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        parameters += step
        if float(np.linalg.norm(step)) < 1e-8:
            break

    sensor_to_tcp = pose_to_matrix(parameters)
    inverse_extrinsic = np.linalg.inv(sensor_to_tcp)
    base_candidates = [
        pose_to_matrix(tcp_pose) @ inverse_extrinsic @ np.linalg.inv(pose_to_matrix(sensor_pose))
        for sensor_pose, tcp_pose in zip(sensor_poses, tcp_poses, strict=True)
    ]
    tracker_to_base = np.eye(4)
    tracker_to_base[:3, :3] = _average_rotations([candidate[:3, :3] for candidate in base_candidates])
    tracker_to_base[:3, 3] = np.mean([candidate[:3, 3] for candidate in base_candidates], axis=0)

    errors = [
        matrix_to_pose(
            np.linalg.inv(pose_to_matrix(tcp_pose))
            @ tracker_to_base
            @ pose_to_matrix(sensor_pose)
            @ sensor_to_tcp
        )
        for sensor_pose, tcp_pose in zip(sensor_poses, tcp_poses, strict=True)
    ]
    position_rmse = float(np.sqrt(np.mean([np.dot(error[:3], error[:3]) for error in errors])))
    orientation_rmse = float(np.sqrt(np.mean([np.dot(error[3:], error[3:]) for error in errors])))
    return tracker_to_base, sensor_to_tcp, position_rmse, orientation_rmse


def save_hand_eye_calibration(
    cfg: DemoConfig,
    tracker_poses: list[np.ndarray],
    tcp_poses: list[np.ndarray],
    tracker_device: Optional[str],
) -> Path:
    tracker_to_base, sensor_to_tcp, position_rmse, orientation_rmse = solve_sensor_to_tcp_hand_eye(
        tracker_poses, tcp_poses
    )
    if position_rmse > cfg.calibration_max_position_rmse_m:
        raise RuntimeError(
            f"Hand-eye calibration position RMSE is {position_rmse:.4f} m, exceeding "
            f"{cfg.calibration_max_position_rmse_m:.4f} m"
        )
    if orientation_rmse > cfg.calibration_max_orientation_rmse_rad:
        raise RuntimeError(
            f"Hand-eye calibration orientation RMSE is {orientation_rmse:.3f} rad, exceeding "
            f"{cfg.calibration_max_orientation_rmse_rad:.3f} rad"
        )
    path = calibration_file(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "tracker_device": tracker_device,
        "pose_pair_count": len(tracker_poses),
        "position_rmse_m": position_rmse,
        "orientation_rmse_rad": orientation_rmse,
        "tracker_to_base_transform": tracker_to_base.tolist(),
        "sensor_to_tcp_transform": sensor_to_tcp.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_rigid_transform(payload: dict, field: str, path: Path) -> np.ndarray:
    transform = np.asarray(payload[field], dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise RuntimeError(f"Invalid {field} in calibration file: {path}")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise RuntimeError(f"{field} is not a rigid transform: {path}")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0:
        raise RuntimeError(f"{field} rotation is invalid: {path}")
    return transform


def load_tracker_to_tcp_calibration(cfg: DemoConfig) -> tuple[np.ndarray, np.ndarray]:
    path = calibration_file(cfg)
    if not path.is_file():
        raise RuntimeError(
            f"Sensor-to-UR calibration is missing: {path}. "
            "Capture at least three paired poses with `ur7e-vla calibrate-demo --config config.yaml`."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Invalid Sensor-to-UR calibration file: {path}") from exc
    if payload.get("version") != 2:
        raise RuntimeError(f"Calibration file {path} uses an obsolete single-pose format; recalibrate with 3+ poses")
    return (
        _load_rigid_transform(payload, "tracker_to_base_transform", path),
        _load_rigid_transform(payload, "sensor_to_tcp_transform", path),
    )


def map_calibrated_tracker_trajectory(
    trajectory: DemonstrationTrajectory,
    tracker_to_base: np.ndarray,
    cfg: DemoConfig,
    sensor_to_tcp: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """Map an absolute Lighthouse trajectory into absolute UR TCP targets."""
    tracker_to_base = np.asarray(tracker_to_base, dtype=np.float64)
    sensor_to_tcp = np.eye(4) if sensor_to_tcp is None else np.asarray(sensor_to_tcp, dtype=np.float64)
    if tracker_to_base.shape != (4, 4) or sensor_to_tcp.shape != (4, 4):
        raise ValueError("calibrated trajectory transforms must both be 4x4")
    tracker_anchor = pose_to_matrix(trajectory.samples[0].tracker_pose)
    tcp_anchor = tracker_to_base @ tracker_anchor @ sensor_to_tcp
    inverse_sensor_to_tcp = np.linalg.inv(sensor_to_tcp)
    targets: list[np.ndarray] = []
    for sample in trajectory.samples:
        delta = np.linalg.inv(tracker_anchor) @ pose_to_matrix(sample.tracker_pose)
        translation = delta[:3, 3] * cfg.translation_scale
        rotation_vector = matrix_to_rotation_vector(delta[:3, :3]) * cfg.rotation_scale
        if np.linalg.norm(translation) > cfg.max_translation_m:
            raise RuntimeError("Demonstration exceeds configured translation workspace limit")
        if np.linalg.norm(rotation_vector) > cfg.max_rotation_rad:
            raise RuntimeError("Demonstration exceeds configured rotation workspace limit")
        scaled_delta = np.eye(4)
        scaled_delta[:3, :3] = rotation_vector_to_matrix(rotation_vector)
        scaled_delta[:3, 3] = translation
        targets.append(matrix_to_pose(tcp_anchor @ inverse_sensor_to_tcp @ scaled_delta @ sensor_to_tcp))
    return targets


def calibrate_demo_sensor_to_tcp(
    cfg: AppConfig, wait_for_operator: Callable[[str], str] = input
) -> Path:
    """Interactively capture multiple shared physical poses, without robot motion.

    The Sensor and TCP samples may be captured sequentially.  The Lighthouse
    base stations and the physical calibration reference must remain fixed
    between the two captures.
    """
    source = PikaSenseSource(cfg.demo)
    robot = UR7e(cfg.robot, execute=False)
    try:
        source.connect()
        robot.connect()
        sensor_poses: list[np.ndarray] = []
        tcp_poses: list[np.ndarray] = []
        count = cfg.demo.calibration_pose_count
        for index in range(count):
            wait_for_operator(
                f"UR TCP {index + 1}/{count}: move TCP to reference pose {index + 1} "
                "with a distinct position and orientation. This command will not move the robot; "
                "keep base stations and reference marks fixed, then press Enter. "
            )
            tcp_poses.append(robot.tcp_pose())
        wait_for_operator(
            "All UR TCP poses captured. Do not move base stations or reference marks. "
            "Prepare to capture Sensor poses in the same reference-pose order, then press Enter. "
        )
        for index in range(count):
            wait_for_operator(
                f"Sensor {index + 1}/{count}: put Pika Sensor at the same physical reference pose {index + 1}. "
                "Keep its position and orientation aligned, then press Enter. "
            )
            sensor_poses.append(source.sample().tracker_pose)
        return save_hand_eye_calibration(cfg.demo, sensor_poses, tcp_poses, source.tracker_device)
    finally:
        try:
            robot.stop()
        finally:
            source.close()


@dataclass(frozen=True)
class PendingFrame:
    image_path: Path
    wrist_image_path: Path
    joints: np.ndarray
    gripper: np.ndarray
    actions: np.ndarray


class PendingEpisode:
    def __init__(self, staging_root: str | Path):
        base = Path(staging_root).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / f"episode_{uuid.uuid4().hex}"
        self.path.mkdir()
        self.frames: list[PendingFrame] = []

    def add(
        self,
        exterior_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        joints: np.ndarray,
        gripper: float,
        action: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        import cv2

        index = len(self.frames)
        image_path = self.path / f"image_{index:06d}.png"
        wrist_path = self.path / f"wrist_{index:06d}.png"
        exterior = cv2.resize(exterior_rgb, (width, height), interpolation=cv2.INTER_AREA)
        wrist = cv2.resize(wrist_rgb, (width, height), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(image_path), cv2.cvtColor(exterior, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to stage {image_path}")
        if not cv2.imwrite(str(wrist_path), cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to stage {wrist_path}")
        self.frames.append(
            PendingFrame(
                image_path=image_path,
                wrist_image_path=wrist_path,
                joints=np.asarray(joints, dtype=np.float32),
                gripper=np.asarray([gripper], dtype=np.float32),
                actions=np.asarray(action, dtype=np.float32),
            )
        )

    def discard(self) -> None:
        if self.path.exists() and self.path.parent != self.path:
            shutil.rmtree(self.path)
        self.frames.clear()


def _read_tasks(path: Path) -> list[str]:
    tasks = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.append(str(json.loads(line)["task"]))
    return tasks


def find_dataset_for_task(root: Path, task: str) -> Optional[Path]:
    wanted = normalize_task(task)
    matches = []
    if root.exists():
        for tasks_file in root.rglob("tasks.jsonl"):
            if any(normalize_task(item) == wanted for item in _read_tasks(tasks_file)):
                matches.append(tasks_file.parent.parent)
    if len(matches) > 1:
        raise RuntimeError(f"Task {task!r} occurs in multiple demo datasets: {matches}")
    return matches[0] if matches else None


def _features(cfg: DemoConfig) -> dict:
    image = {"dtype": "video", "shape": (cfg.image_height, cfg.image_width, 3), "names": ["height", "width", "channel"]}
    return {
        "joints": {"dtype": "float32", "shape": (6,), "names": [f"joint_{i}" for i in range(6)]},
        "gripper": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": [*[f"joint_{i}" for i in range(6)], "gripper"]},
        "image": image,
        "wrist_image": dict(image),
    }


def save_pending_episode(pending: PendingEpisode, task: str, cfg: DemoConfig) -> tuple[Path, int]:
    if not pending.frames:
        raise ValueError("Cannot save an empty episode")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "LeRobot 0.3.3 is required. The installed 'lerobot' package does not expose "
            "lerobot.datasets; install this project's demo optional dependencies."
        ) from exc

    root = Path(cfg.root).expanduser().resolve()
    is_dataset_root = (root / "meta" / "info.json").is_file()
    dataset_root = root if is_dataset_root else find_dataset_for_task(root, task)
    if dataset_root is None:
        # An absent configured root is treated as the exact repo root (the
        # default mani_real case). An existing non-dataset directory is treated
        # as a collection containing one dataset directory per task.
        dataset_root = root if not root.exists() else root / normalize_task(task)
        dataset_root.parent.mkdir(parents=True, exist_ok=True)
        if dataset_root.exists():
            raise RuntimeError(f"New demo directory already exists but is not a valid LeRobot dataset: {dataset_root}")
        dataset = LeRobotDataset.create(
            repo_id=cfg.repo_id,
            root=dataset_root,
            fps=cfg.fps,
            robot_type="ur7e_pika",
            features=_features(cfg),
            use_videos=cfg.use_videos,
            image_writer_threads=4,
        )
    else:
        dataset = LeRobotDataset(repo_id=cfg.repo_id, root=dataset_root)
        if dataset.fps != cfg.fps:
            raise RuntimeError(f"Existing dataset FPS is {dataset.fps}, requested {cfg.fps}")
        required = set(_features(cfg))
        if not required.issubset(dataset.features):
            raise RuntimeError(f"Existing dataset schema is incompatible; missing {sorted(required - set(dataset.features))}")
        dataset.start_image_writer(num_threads=4)

    import cv2

    try:
        for frame in pending.frames:
            image_bgr = cv2.imread(str(frame.image_path))
            wrist_bgr = cv2.imread(str(frame.wrist_image_path))
            if image_bgr is None or wrist_bgr is None:
                raise RuntimeError("A staged camera frame is missing or unreadable")
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            wrist = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)
            dataset.add_frame(
                {
                    "image": image,
                    "wrist_image": wrist,
                    "joints": frame.joints,
                    "gripper": frame.gripper,
                    "actions": frame.actions,
                },
                task=task,
            )
        episode_index = dataset.meta.total_episodes
        dataset.save_episode()
        return dataset_root, episode_index
    finally:
        dataset.stop_image_writer()


class DemoReplay:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.robot = UR7e(cfg.robot, execute=True)
        self.gripper = PikaGripper(cfg.gripper, execute=True)
        self.cameras = CameraPair(cfg.cameras)

    def run(
        self,
        trajectory: DemonstrationTrajectory,
        stop_event: threading.Event,
        on_frame: Optional[Callable[[int], None]] = None,
    ) -> PendingEpisode:
        pending = PendingEpisode(self.cfg.demo.staging_dir)
        period = 1.0 / self.cfg.demo.fps
        try:
            self.robot.connect()
            self.gripper.connect()
            self.cameras.start()
            current_joints = self.robot.joints()
            tracker_to_base, sensor_to_tcp = load_tracker_to_tcp_calibration(self.cfg.demo)
            tcp_targets = map_calibrated_tracker_trajectory(
                trajectory, tracker_to_base, self.cfg.demo, sensor_to_tcp
            )
            joint_targets = []
            near = current_joints
            max_step = min(self.cfg.demo.max_ik_joint_step_rad, self.cfg.robot.max_joint_step_rad)
            lower = np.asarray(self.cfg.robot.joint_min_rad)
            upper = np.asarray(self.cfg.robot.joint_max_rad)
            for tcp_target in tcp_targets:
                target = self.robot.inverse_kinematics(
                    tcp_target,
                    near,
                    self.cfg.demo.ik_position_tolerance_m,
                    self.cfg.demo.ik_orientation_tolerance_rad,
                )
                # The first target is reached with a blocking, slow moveJ below;
                # every recorded replay step still has to satisfy servoJ limits.
                if joint_targets and float(np.max(np.abs(target - near))) > max_step:
                    raise RuntimeError(
                        "IK trajectory contains an unsafe joint discontinuity; slow down the demonstration "
                        "or increase demo FPS"
                    )
                if np.any(target < lower) or np.any(target > upper):
                    raise RuntimeError("IK trajectory violates configured UR joint limits")
                joint_targets.append(target)
                near = target

            previous_gripper = self.gripper.position()
            gripper_targets = [
                self.cfg.gripper.policy_min
                + sample.gripper * (self.cfg.gripper.policy_max - self.cfg.gripper.policy_min)
                for sample in trajectory.samples
            ]
            for index, gripper_target in enumerate(gripper_targets):
                if index and abs(gripper_target - previous_gripper) > self.cfg.demo.max_gripper_step:
                    raise RuntimeError(
                        "Demonstration contains an unsafe gripper step; close/open the Pika Sense more slowly"
                    )
                previous_gripper = gripper_target

            if stop_event.is_set():
                raise RuntimeError("Replay cancelled before moving to start pose")
            LOG.info("Moving safely to calibrated demonstration start pose before recording")
            self.robot.move_joints(
                joint_targets[0],
                self.cfg.demo.approach_joint_speed_rad_s,
                self.cfg.demo.approach_joint_acceleration_rad_s2,
            )
            reached = self.robot.joints()
            if float(np.max(np.abs(reached - joint_targets[0]))) > self.cfg.initial_state.joint_tolerance_rad:
                raise RuntimeError("UR did not reach the calibrated demonstration start pose")
            self.gripper.apply(np.asarray([*joint_targets[0], gripper_targets[0]]))
            if self.cfg.demo.approach_settle_s:
                stop_event.wait(self.cfg.demo.approach_settle_s)
            if stop_event.is_set():
                raise RuntimeError("Replay cancelled before trajectory execution")

            next_tick = time.monotonic()
            for index, (target, gripper_action) in enumerate(zip(joint_targets, gripper_targets, strict=True)):
                if stop_event.is_set():
                    raise RuntimeError("Replay cancelled; episode was not saved")
                joints = self.robot.joints()
                gripper_state = self.gripper.position()
                exterior, wrist = self.cameras.frames()
                action = np.concatenate([target, np.asarray([gripper_action])])
                pending.add(
                    exterior,
                    wrist,
                    joints,
                    gripper_state,
                    action,
                    self.cfg.demo.image_width,
                    self.cfg.demo.image_height,
                )
                self.robot.send_target(target, period_s=period)
                self.gripper.apply(action)
                if on_frame is not None:
                    on_frame(index + 1)
                next_tick += period
                time.sleep(max(0.0, next_tick - time.monotonic()))
            self.robot.stop_servo()
            return pending
        except BaseException:
            pending.discard()
            raise
        finally:
            try:
                self.robot.stop()
            finally:
                try:
                    self.gripper.stop()
                finally:
                    self.cameras.stop()
