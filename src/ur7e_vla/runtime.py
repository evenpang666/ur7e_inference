from __future__ import annotations

import concurrent.futures
import logging
import math
import time
from typing import Optional, Sequence

import numpy as np

from .cameras import CameraPair, resize_with_pad
from .config import AppConfig, TaskStep
from .hardware import PikaGripper, UR7e
from .initial_state import load_task_initial_state, smoothstep
from .policy import OpenPIPolicy
from .recording import VideoRecorder

LOG = logging.getLogger(__name__)


class VLARuntime:
    def __init__(self, cfg: AppConfig, execute: bool = False):
        self.cfg = cfg
        self.execute = execute
        self.robot = UR7e(cfg.robot, execute)
        self.gripper = PikaGripper(cfg.gripper, execute)
        self.cameras = CameraPair(cfg.cameras)
        self.policy: Optional[OpenPIPolicy] = None
        self.recorder: Optional[VideoRecorder] = None
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def _observation(self, task: str) -> dict:
        exterior, wrist = self.cameras.frames()
        joints = self.robot.joints()
        pcfg = self.cfg.policy
        return {
            pcfg.exterior_image_key: resize_with_pad(exterior, pcfg.image_size),
            pcfg.wrist_image_key: resize_with_pad(wrist, pcfg.image_size),
            pcfg.joint_state_key: joints.astype(np.float32),
            pcfg.gripper_state_key: np.asarray([self.gripper.position()], dtype=np.float32),
            "prompt": task,
        }

    def _infer(self, observation: dict) -> tuple[np.ndarray, float]:
        if self.policy is None:
            raise RuntimeError("Policy is not connected")
        inference_start = time.monotonic()
        actions = self.policy.infer(observation)
        inference_latency = time.monotonic() - inference_start
        if inference_latency > self.cfg.policy.max_inference_latency_s:
            raise TimeoutError(
                f"Inference latency {inference_latency:.3f}s exceeded "
                f"{self.cfg.policy.max_inference_latency_s:.3f}s safety limit"
            )
        return actions, inference_latency

    def _restore_initial_state(self, task: str) -> bool:
        cfg = self.cfg.initial_state
        current_joints = self.robot.joints()
        current_gripper = self.gripper.position()
        selected = load_task_initial_state(cfg.path, task, current_joints, cfg.episode)

        lower = np.asarray(self.cfg.robot.joint_min_rad, dtype=np.float64)
        upper = np.asarray(self.cfg.robot.joint_max_rad, dtype=np.float64)
        if np.any(selected.joints < lower) or np.any(selected.joints > upper):
            raise RuntimeError(
                f"Initial state {selected.task_key}/{selected.episode} violates configured joint limits"
            )
        if not self.cfg.gripper.policy_min <= selected.gripper <= self.cfg.gripper.policy_max:
            raise RuntimeError(
                f"Initial gripper value {selected.gripper} is outside configured policy range "
                f"[{self.cfg.gripper.policy_min}, {self.cfg.gripper.policy_max}]"
            )
        if selected.robot_type and selected.robot_type.lower() not in {"ur7e", "ur7"}:
            LOG.warning(
                "Initial-state file declares robot_type=%s; applying its validated 6-joint state to UR7e",
                selected.robot_type,
            )

        LOG.warning(
            "Selected initial state %s/%s; joints=%s, gripper=%.5f",
            selected.task_key,
            selected.episode,
            np.round(selected.joints, 5).tolist(),
            selected.gripper,
        )
        if not self.execute:
            LOG.info("DRY RUN: initial state validated but no restoration motion was sent")
            return True

        joint_distance = float(np.max(np.abs(selected.joints - current_joints)))
        gripper_distance = abs(selected.gripper - current_gripper)
        # Cubic smoothstep reaches 1.5x its average velocity at the midpoint.
        duration_s = 1.5 * max(
            joint_distance / cfg.max_joint_speed_rad_s,
            gripper_distance / cfg.max_gripper_speed_per_s,
        )
        period = 1.0 / cfg.interpolation_hz
        steps = max(1, math.ceil(duration_s * cfg.interpolation_hz))
        LOG.warning(
            "Restoring initial state over %.2fs (%d steps at %.1f Hz)",
            steps * period,
            steps,
            cfg.interpolation_hz,
        )

        for step in range(1, steps + 1):
            if self._stop:
                LOG.warning("Initial-state restoration interrupted before VLA inference")
                self.robot.stop_servo()
                return False
            tick = time.monotonic()
            blend = smoothstep(step / steps)
            joint_target = current_joints + blend * (selected.joints - current_joints)
            gripper_target = current_gripper + blend * (selected.gripper - current_gripper)
            self.robot.send_target(joint_target, period_s=period)
            action = np.concatenate([joint_target, np.asarray([gripper_target])])
            self.gripper.apply(action)
            remaining = period - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)

        settle_steps = math.ceil(cfg.settle_s * cfg.interpolation_hz)
        final_action = np.concatenate([selected.joints, np.asarray([selected.gripper])])
        for _ in range(settle_steps):
            if self._stop:
                LOG.warning("Initial-state settling interrupted before VLA inference")
                self.robot.stop_servo()
                return False
            tick = time.monotonic()
            self.robot.send_target(selected.joints, period_s=period)
            self.gripper.apply(final_action)
            remaining = period - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)
        self.robot.stop_servo()
        actual = self.robot.joints()
        error = float(np.max(np.abs(actual - selected.joints)))
        if error > cfg.joint_tolerance_rad:
            raise RuntimeError(
                f"Initial-state restoration error {error:.4f} rad exceeds "
                f"tolerance {cfg.joint_tolerance_rad:.4f} rad"
            )
        LOG.info("Initial-state restoration complete; max joint error=%.5f rad", error)
        return True

    @staticmethod
    def _keep_running(stop: bool, deadline: Optional[float]) -> bool:
        return not stop and (deadline is None or time.monotonic() < deadline)

    def _log_inference(self, result: tuple[np.ndarray, float]) -> np.ndarray:
        actions, inference_latency = result
        LOG.info(
            "Inference %.3fs, chunk=%s, first_action=%s",
            inference_latency,
            actions.shape,
            np.round(actions[0], 5).tolist(),
        )
        return actions

    @staticmethod
    def _align_actions(actions: np.ndarray, elapsed_control_steps: int) -> np.ndarray:
        """Drop action samples whose 15 Hz control times have already passed."""
        if elapsed_control_steps < 0:
            raise ValueError("elapsed_control_steps cannot be negative")
        if elapsed_control_steps == 0:
            return actions
        if elapsed_control_steps >= actions.shape[0]:
            raise RuntimeError(
                f"Inference result is fully stale: {elapsed_control_steps} control steps elapsed "
                f"for a {actions.shape[0]}-step action chunk"
            )
        LOG.info(
            "Discarding %d stale action(s); %d action(s) remain",
            elapsed_control_steps,
            actions.shape[0] - elapsed_control_steps,
        )
        return actions[elapsed_control_steps:]

    def _run_connected_task(
        self,
        task: str,
        duration_s: float,
        executor: concurrent.futures.ThreadPoolExecutor,
    ) -> tuple[int, int]:
        """Run one prompt while hardware and policy connections remain open."""
        execution_start = None
        action_count = 0
        hold_count = 0
        control_step_count = 0
        mode = "LIVE EXECUTION" if self.execute else "DRY RUN"
        # Motion starts only after a prompt-specific chunk is ready.  No action
        # chunk is carried across an atomic-task boundary.
        LOG.info(
            "Prefetching first policy action chunk for task %r; no initial-state restoration in this step",
            task,
        )
        initial_future = executor.submit(self._infer, self._observation(task))
        current_actions = self._log_inference(initial_future.result())
        execution_start = time.monotonic()
        deadline = execution_start + duration_s
        LOG.warning(
            "Task started in %s mode; task=%r, duration=%.2fs, action_hz=%.1f, steps_per_chunk=%d",
            mode, task, duration_s, self.cfg.robot.control_hz, self.cfg.policy.execute_steps_per_inference,
        )
        period = 1.0 / self.cfg.robot.control_hz
        last_target: Optional[np.ndarray] = None
        while self._keep_running(self._stop, deadline):
            steps = min(self.cfg.policy.execute_steps_per_inference, current_actions.shape[0])
            inference_trigger_step = max(1, steps // 2)
            next_future = None
            next_observation_step: Optional[int] = None
            for step_index, action in enumerate(current_actions[:steps], start=1):
                if not self._keep_running(self._stop, deadline):
                    break
                if self.recorder is not None:
                    self.recorder.check()
                tick = time.monotonic()
                current = self.robot.joints()
                target = self.robot.action_to_target(action, current)
                self.robot.send_target(target)
                self.gripper.apply(action)
                last_target = target
                action_count += 1
                control_step_count += 1
                if next_future is None and step_index >= inference_trigger_step:
                    observation = self._observation(task)
                    next_observation_step = control_step_count
                    next_future = executor.submit(self._infer, observation)
                if action_count % self.cfg.runtime.log_every_n_actions == 0:
                    LOG.info("Task %r applied %d actions; target=%s", task, action_count, np.round(target, 4).tolist())
                remaining = period - (time.monotonic() - tick)
                if remaining > 0:
                    time.sleep(remaining)

            if not self._keep_running(self._stop, deadline):
                break
            if next_future is None:
                observation = self._observation(task)
                next_observation_step = control_step_count
                next_future = executor.submit(self._infer, observation)
            while not next_future.done() and self._keep_running(self._stop, deadline):
                if self.recorder is not None:
                    self.recorder.check()
                tick = time.monotonic()
                if last_target is not None:
                    self.robot.send_target(last_target)
                    hold_count += 1
                    control_step_count += 1
                remaining = period - (time.monotonic() - tick)
                if remaining > 0:
                    time.sleep(remaining)
            if not self._keep_running(self._stop, deadline):
                break
            if next_observation_step is None:
                raise RuntimeError("Missing control-step timestamp for inference request")
            next_actions = self._log_inference(next_future.result())
            current_actions = self._align_actions(next_actions, control_step_count - next_observation_step)
        LOG.info("Task %r finished after %.2fs, %d action steps, %d hold steps", task, time.monotonic() - execution_start, action_count, hold_count)
        return action_count, hold_count

    def run(self, task: str, duration_s: Optional[float] = None) -> None:
        if duration_s is not None and duration_s <= 0:
            raise ValueError("duration must be positive or omitted")
        if duration_s is None:
            # Retain the established unlimited single-task mode.
            return self._run_tasks([(task, None)])
        self._run_tasks([(task, duration_s)])

    def run_sequence(self, steps: Sequence[TaskStep]) -> None:
        if not steps:
            raise ValueError("Task sequence cannot be empty")
        for index, step in enumerate(steps, start=1):
            if not step.task.strip() or step.duration_s <= 0:
                raise ValueError(f"Invalid task sequence step {index}: task must be non-empty and duration positive")
        self._run_tasks([(step.task, step.duration_s) for step in steps])

    def _run_tasks(self, steps: Sequence[tuple[str, Optional[float]]]) -> None:
        invocation_start = time.monotonic()
        executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        total_actions = 0
        total_holds = 0
        try:
            LOG.info("Connecting to UR7e receive interface at %s", self.cfg.robot.ip)
            self.robot.connect()
            self.gripper.connect()
            self.cameras.start()
            if self.cfg.runtime.record_video:
                self.recorder = VideoRecorder(
                    self.cameras.frames,
                    self.cfg.runtime.recording_dir,
                    self.cfg.runtime.recording_fps,
                )
                self.recorder.start()
            self.policy = OpenPIPolicy(self.cfg.policy)
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-inference")
            for index, (task, duration_s) in enumerate(steps, start=1):
                if self._stop:
                    break
                restore_this_task = self.cfg.runtime.restore_before_each_task or (
                    index == 1 and self.cfg.initial_state.enabled
                )
                if restore_this_task:
                    LOG.warning("Restoring JSON initial state before task %d/%d: %r", index, len(steps), task)
                    if not self._restore_initial_state(task):
                        break
                if duration_s is None:
                    # Only the legacy single-task API can be unbounded.
                    duration_s = float("inf")
                LOG.warning("Starting task %d/%d: %r", index, len(steps), task)
                actions, holds = self._run_connected_task(task, duration_s, executor)
                total_actions += actions
                total_holds += holds
        finally:
            # Stop motion first, then release sensors and serial devices.
            try:
                self.robot.stop()
            finally:
                try:
                    self.gripper.stop()
                finally:
                    try:
                        if self.recorder is not None:
                            self.recorder.stop()
                    finally:
                        try:
                            self.cameras.stop()
                        finally:
                            if executor is not None:
                                executor.shutdown(wait=False, cancel_futures=True)
                            LOG.info(
                                "Runtime stopped after %.2fs, %d action steps, %d hold steps",
                                time.monotonic() - invocation_start, total_actions, total_holds,
                            )
