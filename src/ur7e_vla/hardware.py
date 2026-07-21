from __future__ import annotations

import logging

import numpy as np

from .config import GripperConfig, RobotConfig

LOG = logging.getLogger(__name__)


class UR7e:
    def __init__(self, cfg: RobotConfig, execute: bool):
        self.cfg = cfg
        self.execute = execute
        self._control = None
        self._receive = None

    def connect(self) -> None:
        import rtde_receive

        self._receive = rtde_receive.RTDEReceiveInterface(self.cfg.ip)
        if self.execute:
            import rtde_control

            self._control = rtde_control.RTDEControlInterface(self.cfg.ip)

    def joints(self) -> np.ndarray:
        if self._receive is None:
            raise RuntimeError("UR receive interface is not connected")
        values = np.asarray(self._receive.getActualQ(), dtype=np.float64)
        if values.shape != (6,):
            raise RuntimeError(f"Expected 6 UR joints, got {values.shape}")
        return values

    def tcp_pose(self) -> np.ndarray:
        if self._receive is None:
            raise RuntimeError("UR receive interface is not connected")
        values = np.asarray(self._receive.getActualTCPPose(), dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"Expected a finite 6D UR TCP pose, got {values}")
        return values

    def inverse_kinematics(
        self,
        tcp_pose: np.ndarray,
        near: np.ndarray,
        max_position_error: float = 1e-6,
        max_orientation_error: float = 1e-6,
    ) -> np.ndarray:
        if self._control is None:
            raise RuntimeError("UR control interface is not connected")
        tcp_pose = np.asarray(tcp_pose, dtype=np.float64)
        near = np.asarray(near, dtype=np.float64)
        if tcp_pose.shape != (6,) or near.shape != (6,):
            raise ValueError("UR inverse kinematics requires 6D pose and 6D near joints")
        result = np.asarray(
            self._control.getInverseKinematics(
                tcp_pose.tolist(),
                near.tolist(),
                max_position_error,
                max_orientation_error,
            ),
            dtype=np.float64,
        )
        if result.shape != (6,) or not np.all(np.isfinite(result)):
            raise RuntimeError(f"UR inverse kinematics failed for TCP pose {tcp_pose.tolist()}")
        return result

    def action_to_target(self, action: np.ndarray, current: np.ndarray) -> np.ndarray:
        if action.ndim != 1 or not np.all(np.isfinite(action)):
            raise ValueError("Policy action must be a finite one-dimensional vector")
        indices = np.asarray(self.cfg.action_joint_indices)
        if action.size <= int(indices.max()):
            raise ValueError(f"Action has {action.size} dims but mapping needs index {indices.max()}")
        requested = action[indices]
        if self.cfg.action_mode == "joint_delta":
            delta = requested * self.cfg.joint_delta_scale
            delta = np.clip(delta, -self.cfg.max_joint_step_rad, self.cfg.max_joint_step_rad)
            target = current + delta
        else:
            # Absolute policies are still rate-limited to avoid discontinuous jumps.
            delta = np.clip(requested - current, -self.cfg.max_joint_step_rad, self.cfg.max_joint_step_rad)
            target = current + delta
        lower = np.asarray(self.cfg.joint_min_rad)
        upper = np.asarray(self.cfg.joint_max_rad)
        if np.any(target < lower) or np.any(target > upper):
            raise RuntimeError(f"Target violates configured joint limits: {target.tolist()}")
        return target

    def send_target(self, target: np.ndarray, period_s: float | None = None) -> None:
        if not self.execute:
            return
        if self._control is None:
            raise RuntimeError("UR control interface is not connected")
        period = period_s if period_s is not None else 1.0 / self.cfg.control_hz
        if period <= 0:
            raise ValueError("servoJ period must be positive")
        ok = self._control.servoJ(
            target.tolist(),
            self.cfg.max_joint_speed_rad_s,
            1.0,
            period,
            self.cfg.servo_lookahead_s,
            self.cfg.servo_gain,
        )
        if ok is False:
            raise RuntimeError("UR servoJ rejected target")

    def move_joints(self, target: np.ndarray, speed_rad_s: float, acceleration_rad_s2: float) -> None:
        """Blocking joint move used only to reach a verified replay start pose."""
        if not self.execute:
            return
        if self._control is None:
            raise RuntimeError("UR control interface is not connected")
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise ValueError("moveJ requires six finite joint angles")
        if speed_rad_s <= 0 or acceleration_rad_s2 <= 0:
            raise ValueError("moveJ speed and acceleration must be positive")
        lower = np.asarray(self.cfg.joint_min_rad)
        upper = np.asarray(self.cfg.joint_max_rad)
        if np.any(target < lower) or np.any(target > upper):
            raise RuntimeError("moveJ target violates configured joint limits")
        ok = self._control.moveJ(target.tolist(), speed_rad_s, acceleration_rad_s2, False)
        if ok is False:
            raise RuntimeError("UR moveJ rejected replay start target")

    def stop_servo(self) -> None:
        if self._control is not None:
            self._control.servoStop()

    def stop(self) -> None:
        if self._control is not None:
            try:
                self.stop_servo()
            finally:
                self._control.disconnect()
        if self._receive is not None:
            self._receive.disconnect()


class PikaGripper:
    def __init__(self, cfg: GripperConfig, execute: bool):
        self.cfg = cfg
        self.execute = execute
        self._device = None
        self._position = 0.0

    def connect(self) -> None:
        if not self.cfg.enabled:
            return
        try:
            from pika.gripper import Gripper
        except ImportError as exc:
            raise RuntimeError("Pika SDK missing; install agilexrobotics/pika_sdk") from exc
        self._device = Gripper(self.cfg.serial_port)
        if not self._device.connect():
            raise RuntimeError(f"Cannot connect to Pika gripper at {self.cfg.serial_port}")
        if self.execute and not self._device.enable():
            raise RuntimeError("Pika gripper motor enable failed")

    def position(self) -> float:
        if self._device is not None:
            angle = float(self._device.get_motor_position())
            span = self.cfg.max_angle_rad - self.cfg.min_angle_rad
            if span <= 0:
                raise ValueError("gripper.max_angle_rad must exceed min_angle_rad")
            physical = float(np.clip((angle - self.cfg.min_angle_rad) / span, 0.0, 1.0))
            if self.cfg.invert:
                physical = 1.0 - physical
            policy_span = self.cfg.policy_max - self.cfg.policy_min
            self._position = self.cfg.policy_min + physical * policy_span
        return self._position

    def apply(self, action: np.ndarray) -> None:
        if not self.cfg.enabled:
            return
        if action.size <= self.cfg.action_index:
            raise ValueError(f"Action lacks gripper index {self.cfg.action_index}")
        policy_value = float(np.clip(action[self.cfg.action_index], self.cfg.policy_min, self.cfg.policy_max))
        self._position = policy_value
        physical = (policy_value - self.cfg.policy_min) / (self.cfg.policy_max - self.cfg.policy_min)
        if self.cfg.invert:
            physical = 1.0 - physical
        angle = self.cfg.min_angle_rad + physical * (self.cfg.max_angle_rad - self.cfg.min_angle_rad)
        if self.execute:
            if self._device is None:
                raise RuntimeError("Pika gripper is not connected")
            if self._device.set_motor_angle(angle) is False:
                raise RuntimeError("Pika gripper rejected target angle")

    def stop(self) -> None:
        if self._device is not None:
            disable = getattr(self._device, "disable", None)
            if self.execute and callable(disable):
                try:
                    disable()
                except Exception:
                    LOG.exception("Failed to disable Pika gripper")
            try:
                self._device.disconnect()
            except Exception:
                LOG.exception("Failed to disconnect Pika gripper")
