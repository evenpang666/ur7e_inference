"""Passively measure Pika Sense AS5047 encoder responsiveness.

This tool opens only the configured Sensor serial port.  It does not start
Lighthouse tracking, connect to the robot, or send any command to either Pika
device.  Stop demo collection before running it because COM9 cannot be shared.
"""

from __future__ import annotations

import argparse
import time

from ur7e_vla.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--threshold-rad", type=float, default=0.01)
    args = parser.parse_args()
    if args.duration <= 0 or args.threshold_rad <= 0:
        parser.error("--duration and --threshold-rad must be positive")

    from pika.sense import Sense

    cfg = load_config(args.config).demo
    sense = Sense(cfg.pika_sense_port)
    if not sense.connect():
        raise RuntimeError(f"Cannot connect to Pika Sense at {cfg.pika_sense_port}")
    try:
        started = time.monotonic()
        previous = None
        changes = 0
        print(f"CONNECTED port={cfg.pika_sense_port}; move the Sensor open/closed now")
        while (elapsed := time.monotonic() - started) < args.duration:
            data = sense.get_encoder_data()
            raw = float(data.get("rad", float("nan")))
            if previous is None or abs(raw - previous) >= args.threshold_rad:
                print(f"t={elapsed:6.3f}s raw_rad={raw:.4f} angle={data.get('angle', '?')}")
                previous = raw
                changes += 1
            time.sleep(0.01)
        print(f"SUMMARY changes={changes} duration_s={args.duration:.1f}")
    finally:
        sense.disconnect()


if __name__ == "__main__":
    main()
