from __future__ import annotations

import argparse
import logging
import signal

from .cameras import list_opencv_cameras
from .config import TaskStep, load_config
from .runtime import VLARuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UR7e + Pika + D435i remote OpenPI runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run policy inference loop")
    run.add_argument("--config", default="config.yaml")
    run.add_argument("--task", help="override task prompt")
    run.add_argument("--duration", type=float, help="seconds; omitted/null means unlimited")
    run.add_argument("--execute", action="store_true", help="actually command UR7e and gripper")
    run.add_argument("--record-video", action="store_true", help="record both camera views to MP4")
    run.add_argument("--recording-dir", help="override the video output directory")
    run.add_argument(
        "--restore-initial-state",
        action="store_true",
        help="restore the task initial state before policy inference",
    )
    run.add_argument("--initial-state-file", help="override initial_robot_states.json path")
    run.add_argument("--initial-state-episode", help="select an episode instead of the nearest state")
    sequence = sub.add_parser("run-sequence", help="run bounded VLA task prompts in order")
    sequence.add_argument("--config", default="config.yaml")
    sequence.add_argument(
        "--step",
        action="append",
        nargs=2,
        metavar=("TASK", "SECONDS"),
        help="one atomic task and its required positive duration; repeat for every step",
    )
    sequence.add_argument("--execute", action="store_true", help="actually command UR7e and gripper")
    sequence.add_argument("--record-video", action="store_true", help="record both camera views to one MP4")
    sequence.add_argument("--recording-dir", help="override the video output directory")
    sequence.add_argument(
        "--restore-initial-state",
        action="store_true",
        help="restore only the first task's initial state before the sequence",
    )
    sequence.add_argument(
        "--restore-each-task-initial-state",
        action="store_true",
        help="restore every task's JSON initial state before that task starts",
    )
    sequence.add_argument("--initial-state-file", help="override initial_robot_states.json path")
    sequence.add_argument("--initial-state-episode", help="select an episode for the first task")
    sub.add_parser("list-cameras", help="probe local OpenCV/UVC camera indices")
    demo = sub.add_parser("collect-demo", help="record Pika Sense demonstrations into a LeRobot dataset")
    demo.add_argument("--config", default="config.yaml")
    demo.add_argument("--task", required=True, help="natural-language task, for example: pick cube")
    demo.add_argument("--demo-root", help="override local LeRobot demo search/create root")
    demo.add_argument("--execute", action="store_true", help="enable physical UR7e path replay")
    calibrate = sub.add_parser("calibrate-demo", help="calibrate absolute Pika Sensor pose to the UR TCP")
    calibrate.add_argument("--config", default="config.yaml")
    calibrate.add_argument("--poses", type=int, help="number of Sensor/TCP pose pairs; at least 3")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "list-cameras":
        print("OpenCV camera indices:", list_opencv_cameras())
        return
    if args.command == "collect-demo":
        from .demo_gui import launch_demo_gui

        cfg = load_config(args.config)
        if args.demo_root:
            cfg.demo.root = args.demo_root
        cfg.validate()
        launch_demo_gui(cfg, args.task, args.execute)
        return
    if args.command == "calibrate-demo":
        from .demo_collection import calibrate_demo_sensor_to_tcp

        cfg = load_config(args.config)
        if args.poses is not None:
            cfg.demo.calibration_pose_count = args.poses
        cfg.validate()
        path = calibrate_demo_sensor_to_tcp(cfg)
        print(f"Saved Pika Sensor-to-UR TCP calibration: {path}")
        return
    cfg = load_config(args.config)
    if args.record_video:
        cfg.runtime.record_video = True
    if args.recording_dir:
        cfg.runtime.recording_dir = args.recording_dir
    if args.restore_initial_state:
        cfg.initial_state.enabled = True
    if args.command == "run-sequence" and args.restore_each_task_initial_state:
        cfg.runtime.restore_before_each_task = True
    if args.initial_state_file:
        cfg.initial_state.path = args.initial_state_file
    if args.initial_state_episode:
        cfg.initial_state.episode = args.initial_state_episode
    if args.command == "run-sequence" and args.step:
        try:
            cfg.runtime.sequence = [TaskStep(task=task, duration_s=float(seconds)) for task, seconds in args.step]
        except ValueError as exc:
            raise ValueError("--step SECONDS must be a number") from exc
    cfg.validate()
    if args.command == "run-sequence":
        if not cfg.runtime.sequence:
            raise ValueError("run-sequence requires at least one --step TASK SECONDS or runtime.sequence entry")
        runtime = VLARuntime(cfg, execute=args.execute)
        signal.signal(signal.SIGINT, lambda *_: runtime.request_stop())
        signal.signal(signal.SIGTERM, lambda *_: runtime.request_stop())
        runtime.run_sequence(cfg.runtime.sequence)
        return
    task = args.task or cfg.runtime.task
    duration = args.duration if args.duration is not None else cfg.runtime.duration_s
    runtime = VLARuntime(cfg, execute=args.execute)
    signal.signal(signal.SIGINT, lambda *_: runtime.request_stop())
    signal.signal(signal.SIGTERM, lambda *_: runtime.request_stop())
    runtime.run(task=task, duration_s=duration)


if __name__ == "__main__":
    main()
