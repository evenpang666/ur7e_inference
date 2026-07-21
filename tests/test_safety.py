import time

import numpy as np
import pytest

from ur7e_vla.config import AppConfig, GripperConfig, RobotConfig, TaskStep, load_config
from ur7e_vla.cli import build_parser
from ur7e_vla.hardware import PikaGripper, UR7e
from ur7e_vla.policy import OpenPIPolicy
from ur7e_vla.runtime import VLARuntime


def test_dataset_aligned_runtime_defaults():
    cfg = AppConfig()
    assert cfg.robot.control_hz == 15.0
    assert cfg.policy.execute_steps_per_inference == 10


def test_long_sequence_config_requires_bounded_nonempty_steps(tmp_path):
    path = tmp_path / "sequence.yaml"
    path.write_text(
        "runtime:\n"
        "  sequence:\n"
        "    - task: open lid\n"
        "      duration_s: 12\n"
        "    - task: place pcr plate\n"
        "      duration_s: 8.5\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.runtime.sequence == [TaskStep("open lid", 12), TaskStep("place pcr plate", 8.5)]


def test_long_sequence_cli_accepts_repeated_task_duration_pairs():
    args = build_parser().parse_args(
        ["run-sequence", "--step", "open lid", "12", "--step", "place pcr plate", "8.5"]
    )
    assert args.step == [["open lid", "12"], ["place pcr plate", "8.5"]]


def test_long_sequence_cli_accepts_restore_each_task_option():
    args = build_parser().parse_args(
        ["run-sequence", "--step", "open lid", "12", "--restore-each-task-initial-state"]
    )
    assert args.restore_each_task_initial_state is True


def test_long_sequence_rejects_missing_duration():
    runtime = VLARuntime(AppConfig(), execute=False)
    with pytest.raises(ValueError, match="duration positive"):
        runtime.run_sequence([TaskStep("open lid", 0)])


def test_long_sequence_restores_each_task_only_when_enabled(monkeypatch):
    class FakeRobot:
        def connect(self):
            pass

        def stop(self):
            pass

    class FakeGripper:
        def connect(self):
            pass

        def stop(self):
            pass

    class FakeCameras:
        def start(self):
            pass

        def stop(self):
            pass

    class FakePolicy:
        def __init__(self, cfg):
            pass

    monkeypatch.setattr("ur7e_vla.runtime.OpenPIPolicy", FakePolicy)
    cfg = AppConfig()
    cfg.runtime.restore_before_each_task = True
    runtime = VLARuntime(cfg, execute=False)
    runtime.robot = FakeRobot()
    runtime.gripper = FakeGripper()
    runtime.cameras = FakeCameras()
    restored = []
    runtime._restore_initial_state = lambda task: restored.append(task) or True
    runtime._run_connected_task = lambda task, duration, executor: (0, 0)

    runtime.run_sequence([TaskStep("open lid", 1), TaskStep("place pcr plate", 1)])

    assert restored == ["open lid", "place pcr plate"]


def test_joint_delta_is_rate_limited():
    robot = UR7e(RobotConfig(action_mode="joint_delta", max_joint_step_rad=0.02), execute=False)
    target = robot.action_to_target(np.ones(7), np.zeros(6))
    np.testing.assert_allclose(target, np.full(6, 0.02))


def test_non_finite_action_is_rejected():
    robot = UR7e(RobotConfig(), execute=False)
    with pytest.raises(ValueError):
        robot.action_to_target(np.array([0, 0, np.nan, 0, 0, 0, 0]), np.zeros(6))


def test_absolute_action_is_also_rate_limited():
    cfg = RobotConfig(action_mode="joint_position", max_joint_step_rad=0.01)
    robot = UR7e(cfg, execute=False)
    target = robot.action_to_target(np.ones(7), np.zeros(6))
    np.testing.assert_allclose(target, np.full(6, 0.01))


def test_gripper_observation_uses_actual_position():
    class FakeGripper:
        def get_motor_position(self):
            return 0.5

    gripper = PikaGripper(GripperConfig(min_angle_rad=0.0, max_angle_rad=1.0), execute=False)
    gripper._device = FakeGripper()
    assert gripper.position() == pytest.approx(0.5)


def test_gripper_maps_policy_units_to_physical_angle():
    class FakeGripper:
        commanded = None

        def set_motor_angle(self, angle):
            self.commanded = angle
            return True

    cfg = GripperConfig(policy_min=-1.0, policy_max=1.0, min_angle_rad=0.2, max_angle_rad=1.2)
    gripper = PikaGripper(cfg, execute=True)
    gripper._device = FakeGripper()
    gripper.apply(np.array([0, 0, 0, 0, 0, 0, 0.0]))
    assert gripper._device.commanded == pytest.approx(0.7)


def test_observation_matches_mani_real_pi05_protocol(monkeypatch):
    monkeypatch.setattr(
        "ur7e_vla.runtime.resize_with_pad",
        lambda image, size: np.zeros((size, size, 3), dtype=np.uint8),
    )
    runtime = VLARuntime(AppConfig(), execute=False)
    runtime.cameras.frames = lambda: (
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.uint8),
    )
    runtime.robot.joints = lambda: np.arange(6, dtype=np.float64)
    runtime.gripper.position = lambda: 0.25

    observation = runtime._observation("close lid")

    assert set(observation) == {
        "observation/image",
        "observation/wrist_image",
        "observation/joints",
        "observation/gripper",
        "prompt",
    }
    assert observation["observation/image"].shape == (224, 224, 3)
    assert observation["observation/image"].dtype == np.uint8
    assert observation["observation/wrist_image"].shape == (224, 224, 3)
    assert observation["observation/joints"].shape == (6,)
    assert observation["observation/joints"].dtype == np.float32
    assert observation["observation/gripper"].shape == (1,)
    assert observation["prompt"] == "close lid"


def test_policy_rejects_non_seven_dimensional_actions():
    class FakeClient:
        def infer(self, observation):
            return {"actions": np.zeros((5, 8))}

    policy = OpenPIPolicy.__new__(OpenPIPolicy)
    policy.cfg = AppConfig().policy
    policy._client = FakeClient()
    with pytest.raises(RuntimeError, match="7 dims"):
        policy.infer({})


def test_async_actions_are_aligned_to_the_observation_control_step():
    actions = np.arange(50 * 7, dtype=np.float64).reshape(50, 7)

    aligned = VLARuntime._align_actions(actions, elapsed_control_steps=5)

    np.testing.assert_array_equal(aligned, actions[5:])


def test_fully_stale_action_chunk_is_rejected():
    actions = np.zeros((5, 7), dtype=np.float64)

    with pytest.raises(RuntimeError, match="fully stale"):
        VLARuntime._align_actions(actions, elapsed_control_steps=5)


def test_async_replace_merges_new_chunk_over_queue_overlap():
    queued = np.zeros((3, 7), dtype=np.float64)
    incoming = np.ones((2, 7), dtype=np.float64)

    merged = VLARuntime._merge_action_queue(queued, incoming, "replace", 0.7, 0.1)

    np.testing.assert_array_equal(merged, np.array([[1] * 7, [1] * 7, [0] * 7], dtype=np.float64))


def test_async_guard_keeps_large_action_changes_but_replaces_small_ones():
    queued = np.zeros((2, 7), dtype=np.float64)
    incoming = np.array([np.full(7, 0.05), np.full(7, 0.2)])

    merged = VLARuntime._merge_action_queue(queued, incoming, "guard", 0.7, 0.1)

    np.testing.assert_allclose(merged[0], incoming[0])
    np.testing.assert_allclose(merged[1], queued[1])


def test_control_keeps_sending_while_next_inference_runs(monkeypatch):
    monkeypatch.setattr(
        "ur7e_vla.runtime.resize_with_pad",
        lambda image, size: np.zeros((size, size, 3), dtype=np.uint8),
    )

    class FakePolicy:
        def __init__(self, cfg):
            pass

        def infer(self, observation):
            time.sleep(0.15)
            return np.zeros((50, 7), dtype=np.float64)

    class FakeRobot:
        def __init__(self):
            self.send_times = []

        def connect(self):
            pass

        def joints(self):
            return np.zeros(6, dtype=np.float64)

        def action_to_target(self, action, current):
            return action[:6]

        def send_target(self, target):
            self.send_times.append(time.monotonic())

        def stop(self):
            pass

    class FakeGripper:
        def connect(self):
            pass

        def position(self):
            return 0.0

        def apply(self, action):
            pass

        def stop(self):
            pass

    class FakeCameras:
        def start(self):
            pass

        def frames(self):
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            return frame, frame

        def stop(self):
            pass

    monkeypatch.setattr("ur7e_vla.runtime.OpenPIPolicy", FakePolicy)
    cfg = AppConfig()
    cfg.robot.control_hz = 50.0
    cfg.policy.execute_steps_per_inference = 10
    runtime = VLARuntime(cfg, execute=False)
    runtime.robot = FakeRobot()
    runtime.gripper = FakeGripper()
    runtime.cameras = FakeCameras()

    runtime.run("test", duration_s=0.45)

    gaps = np.diff(runtime.robot.send_times)
    assert len(gaps) >= 10
    # Sequential inference would create ~0.15 s gaps. The async loop holds the
    # last servo target, keeping gaps close to the configured 0.02 s period.
    assert float(np.max(gaps)) < 0.06
