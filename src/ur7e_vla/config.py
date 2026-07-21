from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml


@dataclass
class PolicyConfig:
    host: str = "192.168.124.15"
    port: int = 8000
    exterior_image_key: str = "observation/image"
    wrist_image_key: str = "observation/wrist_image"
    joint_state_key: str = "observation/joints"
    gripper_state_key: str = "observation/gripper"
    image_size: int = 224
    action_dim: int = 7
    execute_steps_per_inference: int = 10
    max_inference_latency_s: float = 2.0
    inference_mode: str = "asynchronous"
    synchronous_execute_steps: int = 8
    async_queue_trigger_fraction: float = 0.7
    async_merge_mode: str = "weighted_blend"
    async_blend_newest_weight: float = 0.7
    async_guard_max_difference: float = 0.10


@dataclass
class RobotConfig:
    ip: str = "169.254.175.10"
    control_hz: float = 15.0
    action_mode: str = "joint_position"
    action_joint_indices: list[int] = field(default_factory=lambda: list(range(6)))
    joint_state_size: int = 6
    joint_delta_scale: float = 1.0
    max_joint_step_rad: float = 0.02
    max_joint_speed_rad_s: float = 0.30
    servo_lookahead_s: float = 0.10
    servo_gain: int = 300
    joint_min_rad: list[float] = field(default_factory=lambda: [-6.283, -6.283, -3.142, -6.283, -6.283, -6.283])
    joint_max_rad: list[float] = field(default_factory=lambda: [6.283, 6.283, 3.142, 6.283, 6.283, 6.283])


@dataclass
class CameraConfig:
    realsense_serial: Optional[str] = None
    realsense_width: int = 640
    realsense_height: int = 480
    realsense_fps: int = 30
    wrist_realsense_serial: Optional[str] = None
    wrist_device: Union[int, str] = 0
    wrist_width: int = 640
    wrist_height: int = 480
    wrist_fps: int = 30
    max_frame_age_s: float = 0.5


@dataclass
class GripperConfig:
    enabled: bool = True
    serial_port: str = "/dev/ttyUSB1"
    action_index: int = 6
    policy_min: float = 0.0
    policy_max: float = 1.0
    min_angle_rad: float = 0.0
    max_angle_rad: float = 1.0
    invert: bool = False


@dataclass
class InitialStateConfig:
    enabled: bool = False
    path: str = "initial_robot_states.json"
    episode: Optional[str] = None
    interpolation_hz: float = 50.0
    max_joint_speed_rad_s: float = 0.15
    max_gripper_speed_per_s: float = 0.25
    joint_tolerance_rad: float = 0.03
    settle_s: float = 0.25


@dataclass
class RuntimeConfig:
    task: str = "pick up the object"
    duration_s: Optional[float] = None
    log_every_n_actions: int = 20
    record_video: bool = False
    recording_fps: float = 30.0
    recording_dir: str = "recordings"
    # In long-sequence mode, restore each task's JSON initial state before it
    # begins. Disabled by default so tasks naturally continue from prior state.
    restore_before_each_task: bool = False
    sequence: list["TaskStep"] = field(default_factory=list)


@dataclass(frozen=True)
class TaskStep:
    """One bounded prompt in a long VLA task sequence."""

    task: str
    duration_s: float


@dataclass
class DemoConfig:
    root: str = "~/.cache/huggingface/lerobot/mani_real"
    repo_id: str = "mani_real"
    staging_dir: str = "demo_staging"
    fps: int = 50
    image_width: int = 256
    image_height: int = 256
    use_videos: bool = True
    pika_sense_port: str = "COM6"
    tracker_device: Optional[str] = None
    tracker_backend: str = "pysurvive"
    tracker_start_timeout_s: float = 30.0
    max_tracker_age_s: float = 0.25
    # Created by `ur7e-vla calibrate-demo`.  It maps a Lighthouse/Sensor pose
    # directly into the UR TCP frame and is deliberately required for replay.
    calibration_path: str = "pika_sensor_to_ur_tcp.json"
    calibration_pose_count: int = 5
    calibration_max_position_rmse_m: float = 0.03
    calibration_max_orientation_rmse_rad: float = 0.35
    approach_joint_speed_rad_s: float = 0.15
    approach_joint_acceleration_rad_s2: float = 0.30
    approach_settle_s: float = 0.25
    gripper_distance_min_mm: float = 0.0
    gripper_distance_max_mm: float = 85.0
    gripper_invert: bool = False
    translation_scale: float = 1.0
    rotation_scale: float = 1.0
    max_translation_m: float = 0.50
    max_rotation_rad: float = 3.142
    max_ik_joint_step_rad: float = 0.02
    max_gripper_step: float = 0.10
    ik_position_tolerance_m: float = 1e-6
    ik_orientation_tolerance_rad: float = 1e-6


@dataclass
class AppConfig:
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    cameras: CameraConfig = field(default_factory=CameraConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    initial_state: InitialStateConfig = field(default_factory=InitialStateConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)

    def validate(self) -> None:
        if self.robot.action_mode not in {"joint_delta", "joint_position"}:
            raise ValueError("robot.action_mode must be joint_delta or joint_position")
        if self.robot.control_hz <= 0 or self.policy.execute_steps_per_inference <= 0:
            raise ValueError("control_hz and execute_steps_per_inference must be positive")
        if len(self.robot.action_joint_indices) != 6:
            raise ValueError("UR7e requires exactly 6 action_joint_indices")
        if len(set(self.robot.action_joint_indices)) != 6 or min(self.robot.action_joint_indices) < 0:
            raise ValueError("action_joint_indices must be 6 unique non-negative indices")
        if len(self.robot.joint_min_rad) != 6 or len(self.robot.joint_max_rad) != 6:
            raise ValueError("joint_min_rad and joint_max_rad must each contain 6 values")
        if self.robot.joint_state_size != 6:
            raise ValueError("mani_real_pi05 requires exactly 6 UR joint states")
        if self.runtime.duration_s is not None and self.runtime.duration_s <= 0:
            raise ValueError("runtime.duration_s must be null or positive")
        if not isinstance(self.runtime.restore_before_each_task, bool):
            raise ValueError("runtime.restore_before_each_task must be boolean")
        for index, step in enumerate(self.runtime.sequence, start=1):
            if not isinstance(step, TaskStep):
                raise ValueError(f"runtime.sequence[{index}] must be a task/duration mapping")
            if not isinstance(step.task, str) or not step.task.strip():
                raise ValueError(f"runtime.sequence[{index}].task cannot be empty")
            if step.duration_s <= 0:
                raise ValueError(f"runtime.sequence[{index}].duration_s must be positive")
        if self.cameras.max_frame_age_s <= 0:
            raise ValueError("cameras.max_frame_age_s must be positive")
        if self.policy.max_inference_latency_s <= 0:
            raise ValueError("policy.max_inference_latency_s must be positive")
        if self.policy.inference_mode not in {"synchronous", "asynchronous"}:
            raise ValueError("policy.inference_mode must be synchronous or asynchronous")
        if self.policy.synchronous_execute_steps <= 0:
            raise ValueError("policy.synchronous_execute_steps must be positive")
        if not 0 < self.policy.async_queue_trigger_fraction <= 1:
            raise ValueError("policy.async_queue_trigger_fraction must be in (0, 1]")
        if self.policy.async_merge_mode not in {"replace", "weighted_blend", "guard"}:
            raise ValueError("policy.async_merge_mode must be replace, weighted_blend, or guard")
        if not 0 <= self.policy.async_blend_newest_weight <= 1:
            raise ValueError("policy.async_blend_newest_weight must be in [0, 1]")
        if self.policy.async_guard_max_difference < 0:
            raise ValueError("policy.async_guard_max_difference must not be negative")
        if self.policy.action_dim != 7:
            raise ValueError("mani_real_pi05 returns exactly 7 action dimensions")
        if max(self.robot.action_joint_indices + [self.gripper.action_index]) >= self.policy.action_dim:
            raise ValueError("robot/gripper action mapping exceeds policy.action_dim")
        if self.gripper.action_index < 0 or self.gripper.max_angle_rad <= self.gripper.min_angle_rad:
            raise ValueError("invalid gripper action index or physical angle range")
        if self.gripper.policy_max <= self.gripper.policy_min:
            raise ValueError("gripper.policy_max must exceed policy_min")
        if self.runtime.log_every_n_actions <= 0:
            raise ValueError("runtime.log_every_n_actions must be positive")
        if self.runtime.recording_fps <= 0:
            raise ValueError("runtime.recording_fps must be positive")
        if not self.runtime.recording_dir.strip():
            raise ValueError("runtime.recording_dir cannot be empty")
        if not self.initial_state.path.strip():
            raise ValueError("initial_state.path cannot be empty")
        if self.initial_state.interpolation_hz <= 0:
            raise ValueError("initial_state.interpolation_hz must be positive")
        if self.initial_state.max_joint_speed_rad_s <= 0:
            raise ValueError("initial_state.max_joint_speed_rad_s must be positive")
        if self.initial_state.max_gripper_speed_per_s <= 0:
            raise ValueError("initial_state.max_gripper_speed_per_s must be positive")
        if self.initial_state.joint_tolerance_rad <= 0 or self.initial_state.settle_s < 0:
            raise ValueError("invalid initial-state tolerance or settle time")
        if self.demo.fps <= 0 or self.demo.image_width <= 0 or self.demo.image_height <= 0:
            raise ValueError("demo fps and image dimensions must be positive")
        if self.demo.tracker_backend not in {"pysurvive"}:
            raise ValueError("demo.tracker_backend must be pysurvive")
        if self.demo.max_tracker_age_s <= 0 or self.demo.tracker_start_timeout_s <= 0:
            raise ValueError("demo tracker timeouts must be positive")
        if not self.demo.calibration_path.strip():
            raise ValueError("demo.calibration_path cannot be empty")
        if self.demo.calibration_pose_count < 3:
            raise ValueError("demo.calibration_pose_count must be at least 3")
        if self.demo.calibration_max_position_rmse_m <= 0 or self.demo.calibration_max_orientation_rmse_rad <= 0:
            raise ValueError("demo calibration residual limits must be positive")
        if self.demo.approach_joint_speed_rad_s <= 0 or self.demo.approach_joint_acceleration_rad_s2 <= 0:
            raise ValueError("demo approach joint speed and acceleration must be positive")
        if self.demo.approach_settle_s < 0:
            raise ValueError("demo.approach_settle_s must not be negative")
        if self.demo.gripper_distance_max_mm <= self.demo.gripper_distance_min_mm:
            raise ValueError("demo gripper distance range is invalid")
        if self.demo.translation_scale <= 0 or self.demo.rotation_scale <= 0:
            raise ValueError("demo pose scales must be positive")
        if self.demo.max_translation_m <= 0 or self.demo.max_rotation_rad <= 0:
            raise ValueError("demo workspace limits must be positive")
        if self.demo.max_ik_joint_step_rad <= 0:
            raise ValueError("demo.max_ik_joint_step_rad must be positive")
        if self.demo.max_gripper_step <= 0:
            raise ValueError("demo.max_gripper_step must be positive")


def _make(cls: type, values: Optional[dict[str, Any]]):
    return cls(**(values or {}))


def load_config(path: Union[str, Path]) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    runtime_values = dict(raw.get("runtime") or {})
    raw_sequence = runtime_values.pop("sequence", [])
    if not isinstance(raw_sequence, list):
        raise ValueError("runtime.sequence must be a list")
    try:
        sequence = [
            TaskStep(task=item["task"], duration_s=float(item["duration_s"]))
            for item in raw_sequence
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Each runtime.sequence item must contain task and duration_s") from exc
    cfg = AppConfig(
        policy=_make(PolicyConfig, raw.get("policy")),
        robot=_make(RobotConfig, raw.get("robot")),
        cameras=_make(CameraConfig, raw.get("cameras")),
        gripper=_make(GripperConfig, raw.get("gripper")),
        initial_state=_make(InitialStateConfig, raw.get("initial_state")),
        runtime=RuntimeConfig(**runtime_values, sequence=sequence),
        demo=_make(DemoConfig, raw.get("demo")),
    )
    cfg.validate()
    return cfg
