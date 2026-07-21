"""Send one synthetic mani_real_pi05 observation without touching robot hardware."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from openpi_client import websocket_client_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.124.15")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="connectivity test; do not execute")
    args = parser.parse_args()

    started = time.perf_counter()
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    connected = time.perf_counter()
    metadata = client.get_server_metadata()
    print("CONNECTED", f"{args.host}:{args.port}", f"handshake_s={connected - started:.3f}")
    print("METADATA", json.dumps(metadata, ensure_ascii=False, default=str))

    observation = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joints": np.zeros(6, dtype=np.float32),
        "observation/gripper": np.zeros(1, dtype=np.float32),
        "prompt": args.prompt,
    }
    inference_started = time.perf_counter()
    result = client.infer(observation)
    inference_s = time.perf_counter() - inference_started
    if "actions" not in result:
        raise RuntimeError(f"Response has no actions key: {list(result)}")
    actions = np.asarray(result["actions"])
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise RuntimeError(f"Expected actions [chunk, 7], got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("Response contains NaN or infinite actions")
    print("INFERENCE_OK", f"latency_s={inference_s:.3f}", f"shape={actions.shape}", f"dtype={actions.dtype}")
    print("FIRST_ACTION", np.array2string(actions[0], precision=6, separator=","))


if __name__ == "__main__":
    main()
