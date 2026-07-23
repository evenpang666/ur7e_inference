"""Read or deliberately test the configured Pika gripper endpoint.

The default mode only verifies the serial connection and reports the motor
position.  Motion is deliberately opt-in with ``--execute``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ur7e_vla.config import load_config
from ur7e_vla.hardware import PikaGripper


def policy_value_for_angle(gripper_cfg, angle: float) -> float:
    """Convert a physical Pika motor angle to this project's policy units."""
    fraction = (angle - gripper_cfg.min_angle_rad) / (
        gripper_cfg.max_angle_rad - gripper_cfg.min_angle_rad
    )
    if gripper_cfg.invert:
        fraction = 1.0 - fraction
    return gripper_cfg.policy_min + fraction * (
        gripper_cfg.policy_max - gripper_cfg.policy_min
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--target",
        choices=("status", "closed", "open"),
        default="status",
        help="status is read-only; closed/open require --execute.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow a physical command. Ensure the gripper is clear first.",
    )
    parser.add_argument("--settle-s", type=float, default=1.5)
    args = parser.parse_args()

    if args.target != "status" and not args.execute:
        parser.error("Refusing physical motion without --execute")
    if args.settle_s < 0:
        parser.error("--settle-s must be non-negative")

    cfg = load_config(args.config).gripper
    gripper = PikaGripper(cfg, execute=args.execute)
    try:
        gripper.connect()
        before = gripper.position()
        motor_data = getattr(gripper._device, "get_motor_data", lambda: {})()
        print(
            f"CONNECTED port={cfg.serial_port} motor_rad="
            f"{float(gripper._device.get_motor_position()):.4f} policy={before:.4f}"
        )
        print(f"MOTOR_DATA {motor_data}")
        if args.target == "status":
            return

        angle = cfg.min_angle_rad if args.target == "closed" else cfg.max_angle_rad
        policy_value = policy_value_for_angle(cfg, angle)
        action = np.zeros(cfg.action_index + 1, dtype=np.float64)
        action[cfg.action_index] = policy_value
        gripper.apply(action)
        print(
            f"COMMAND target={args.target} policy={policy_value:.4f} "
            f"motor_rad={angle:.4f}; waiting {args.settle_s:.1f}s"
        )
        time.sleep(args.settle_s)
        actual_angle = float(gripper._device.get_motor_position())
        motor_data = getattr(gripper._device, "get_motor_data", lambda: {})()
        print(
            f"READBACK motor_rad={actual_angle:.4f} policy={gripper.position():.4f} "
            f"error_rad={abs(actual_angle - angle):.4f}"
        )
        print(f"MOTOR_DATA {motor_data}")
    finally:
        gripper.stop()


if __name__ == "__main__":
    main()
