from __future__ import annotations

import numpy as np

from .config import PolicyConfig


class OpenPIPolicy:
    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        try:
            from openpi_client import websocket_client_policy
        except ImportError as exc:
            raise RuntimeError(
                "openpi-client is missing. Install it from OPENPI_ROOT/packages/openpi-client"
            ) from exc
        self._client = websocket_client_policy.WebsocketClientPolicy(host=cfg.host, port=cfg.port)

    def infer(self, observation: dict) -> np.ndarray:
        result = self._client.infer(observation)
        if "actions" not in result:
            raise RuntimeError(f"Policy response has no 'actions' key: {result.keys()}")
        actions = np.asarray(result["actions"], dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[0] == 0 or not np.all(np.isfinite(actions)):
            raise RuntimeError(f"Invalid action chunk shape/content: {actions.shape}")
        if actions.shape[1] != self.cfg.action_dim:
            raise RuntimeError(
                f"Expected mani_real_pi05 actions with {self.cfg.action_dim} dims, got {actions.shape}"
            )
        return actions
