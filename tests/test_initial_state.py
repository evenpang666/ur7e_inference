import json

import numpy as np
import pytest

from ur7e_vla.config import AppConfig
from ur7e_vla.initial_state import load_task_initial_state, normalize_task_key, smoothstep
from ur7e_vla.runtime import VLARuntime


def write_states(path):
    payload = {
        "robot_type": "ur7e",
        "state_fields": ["j0", "j1", "j2", "j3", "j4", "j5", "gripper"],
        "tasks": {
            "open_lid": {
                "episode_000000": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9],
                "episode_000001": [0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.8],
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_task_lookup_normalizes_spaces_and_selects_nearest_episode(tmp_path):
    path = tmp_path / "states.json"
    write_states(path)

    state = load_task_initial_state(path, "open lid", np.zeros(6))

    assert normalize_task_key(" Open-Lid ") == "open_lid"
    assert state.task_key == "open_lid"
    assert state.episode == "episode_000001"
    np.testing.assert_allclose(state.joints, [0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    assert state.gripper == pytest.approx(0.8)


def test_task_lookup_can_select_an_explicit_episode(tmp_path):
    path = tmp_path / "states.json"
    write_states(path)

    state = load_task_initial_state(path, "open_lid", np.zeros(6), "episode_000000")

    assert state.episode == "episode_000000"


def test_unknown_task_is_rejected_before_motion(tmp_path):
    path = tmp_path / "states.json"
    write_states(path)

    with pytest.raises(KeyError, match="available tasks: open_lid"):
        load_task_initial_state(path, "press button", np.zeros(6))


def test_smoothstep_has_stationary_endpoints():
    assert smoothstep(0.0) == pytest.approx(0.0)
    assert smoothstep(0.5) == pytest.approx(0.5)
    assert smoothstep(1.0) == pytest.approx(1.0)


def test_runtime_restores_joints_and_gripper_before_inference(tmp_path):
    path = tmp_path / "states.json"
    write_states(path)

    class FakeRobot:
        def __init__(self):
            self.actual = np.zeros(6, dtype=np.float64)
            self.targets = []
            self.servo_stopped = False

        def joints(self):
            return self.actual.copy()

        def send_target(self, target, period_s=None):
            self.actual = np.asarray(target).copy()
            self.targets.append((self.actual.copy(), period_s))

        def stop_servo(self):
            self.servo_stopped = True

    class FakeGripper:
        def __init__(self):
            self.value = 0.0

        def position(self):
            return self.value

        def apply(self, action):
            self.value = float(action[6])

    cfg = AppConfig()
    cfg.initial_state.enabled = True
    cfg.initial_state.path = str(path)
    cfg.initial_state.interpolation_hz = 100.0
    cfg.initial_state.max_joint_speed_rad_s = 1000.0
    cfg.initial_state.max_gripper_speed_per_s = 1000.0
    cfg.initial_state.joint_tolerance_rad = 0.001
    cfg.initial_state.settle_s = 0.02
    runtime = VLARuntime(cfg, execute=True)
    runtime.robot = FakeRobot()
    runtime.gripper = FakeGripper()

    assert runtime._restore_initial_state("open lid") is True

    np.testing.assert_allclose(runtime.robot.actual, [0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    assert runtime.gripper.value == pytest.approx(0.8)
    assert runtime.robot.targets[-1][1] == pytest.approx(0.01)
    assert len(runtime.robot.targets) == 3  # one interpolation step + two final hold steps
    assert runtime.robot.servo_stopped is True
