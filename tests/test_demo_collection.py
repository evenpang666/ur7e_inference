import json
from pathlib import Path

import numpy as np

from ur7e_vla.cli import build_parser
from ur7e_vla.config import DemoConfig
from ur7e_vla.demo_collection import (
    DemonstrationTrajectory,
    PendingEpisode,
    PikaSenseSource,
    SensorSample,
    find_dataset_for_task,
    map_calibrated_tracker_trajectory,
    map_tracker_trajectory,
    matrix_to_pose,
    normalize_task,
    pose_to_matrix,
    solve_sensor_to_tcp_hand_eye,
)


def test_lighthouse_objects_are_not_valid_tracker_candidates():
    assert PikaSenseSource._is_lighthouse("LH0")
    assert PikaSenseSource._is_lighthouse("lh12")
    assert not PikaSenseSource._is_lighthouse("WM0")


def test_collect_demo_cli_requires_explicit_execute():
    args = build_parser().parse_args(["collect-demo", "--task", "pick cube"])
    assert args.task == "pick cube"
    assert args.execute is False


def test_pose_matrix_round_trip():
    pose = np.asarray([0.1, -0.2, 0.3, 0.2, -0.3, 0.4])
    assert np.allclose(matrix_to_pose(pose_to_matrix(pose)), pose, atol=1e-8)


def test_tracker_path_is_relative_to_robot_replay_anchor():
    trajectory = DemonstrationTrajectory(
        samples=(
            SensorSample(0.0, np.asarray([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]), 0.0),
            SensorSample(0.1, np.asarray([1.1, 2.0, 3.0, 0.0, 0.0, 0.2]), 1.0),
        ),
        fps=10,
    )
    targets = map_tracker_trajectory(
        trajectory,
        np.asarray([0.5, 0.0, 0.4, 0.0, 0.0, 0.0]),
        DemoConfig(max_translation_m=0.2),
    )
    assert np.allclose(targets[0], [0.5, 0.0, 0.4, 0.0, 0.0, 0.0], atol=1e-8)
    assert np.allclose(targets[1], [0.6, 0.0, 0.4, 0.0, 0.0, 0.2], atol=1e-8)


def test_calibrated_path_has_an_absolute_start_pose():
    trajectory = DemonstrationTrajectory(
        samples=(
            SensorSample(0.0, np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.0),
            SensorSample(0.1, np.asarray([1.1, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.0),
        ),
        fps=10,
    )
    calibration = pose_to_matrix(np.asarray([-0.8, 0.2, 0.4, 0.0, 0.0, 0.0]))
    targets = map_calibrated_tracker_trajectory(trajectory, calibration, DemoConfig(max_translation_m=0.2))
    assert np.allclose(targets[0], [0.2, 0.2, 0.4, 0.0, 0.0, 0.0], atol=1e-8)
    assert np.allclose(targets[1], [0.3, 0.2, 0.4, 0.0, 0.0, 0.0], atol=1e-8)


def test_multi_pose_hand_eye_recovers_sensor_tcp_extrinsic():
    tracker_to_base = pose_to_matrix(np.asarray([0.3, -0.2, 0.5, 0.2, -0.1, 0.3]))
    sensor_to_tcp = pose_to_matrix(np.asarray([0.12, -0.06, 0.08, -0.25, 0.15, 0.2]))
    sensor_poses = [
        np.asarray([0.1, 0.2, 0.4, 0.1, 0.2, -0.1]),
        np.asarray([0.4, -0.1, 0.5, -0.3, 0.1, 0.4]),
        np.asarray([-0.2, 0.3, 0.7, 0.2, -0.4, 0.1]),
        np.asarray([0.2, 0.6, 0.3, -0.1, 0.5, -0.3]),
    ]
    tcp_poses = [
        matrix_to_pose(tracker_to_base @ pose_to_matrix(sensor_pose) @ sensor_to_tcp)
        for sensor_pose in sensor_poses
    ]
    solved_a, solved_b, position_rmse, orientation_rmse = solve_sensor_to_tcp_hand_eye(sensor_poses, tcp_poses)
    assert position_rmse < 1e-6
    assert orientation_rmse < 1e-6
    for sensor_pose, tcp_pose in zip(sensor_poses, tcp_poses, strict=True):
        assert np.allclose(solved_a @ pose_to_matrix(sensor_pose) @ solved_b, pose_to_matrix(tcp_pose), atol=1e-5)


def test_find_dataset_uses_normalized_task(tmp_path: Path):
    dataset = tmp_path / "existing"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "Pick Cube"}) + "\n",
        encoding="utf-8",
    )
    assert find_dataset_for_task(tmp_path, "pick-cube") == dataset
    assert normalize_task("  Pick   Cube ") == "pick_cube"


def test_pending_episode_discard_is_local_to_staging(tmp_path: Path):
    pending = PendingEpisode(tmp_path)
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    assert pending.path.exists()
    pending.discard()
    assert not pending.path.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"
