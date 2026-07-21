from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TaskInitialState:
    task_key: str
    episode: str
    joints: np.ndarray
    gripper: float
    robot_type: Optional[str]


def normalize_task_key(task: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", task.strip().lower()).strip("_")
    if not key:
        raise ValueError("Task cannot be empty when initial-state restoration is enabled")
    return key


def load_task_initial_state(
    path: str | Path,
    task: str,
    current_joints: np.ndarray,
    episode: Optional[str] = None,
) -> TaskInitialState:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    tasks = payload.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError(f"Initial-state file has no task mapping: {source}")

    normalized_tasks: dict[str, tuple[str, dict]] = {}
    for raw_key, states in tasks.items():
        normalized = normalize_task_key(str(raw_key))
        if normalized in normalized_tasks:
            raise ValueError(f"Duplicate normalized task key {normalized!r} in {source}")
        if not isinstance(states, dict) or not states:
            raise ValueError(f"Task {raw_key!r} has no episode states in {source}")
        normalized_tasks[normalized] = (str(raw_key), states)

    requested_key = normalize_task_key(task)
    if requested_key not in normalized_tasks:
        available = ", ".join(sorted(normalized_tasks))
        raise KeyError(f"No initial state for task {task!r} ({requested_key}); available tasks: {available}")
    task_key, episode_states = normalized_tasks[requested_key]

    parsed: dict[str, np.ndarray] = {}
    for episode_name, values in episode_states.items():
        state = np.asarray(values, dtype=np.float64)
        if state.shape != (7,) or not np.all(np.isfinite(state)):
            raise ValueError(f"Invalid state for {task_key}/{episode_name}: expected 7 finite values")
        parsed[str(episode_name)] = state

    if episode is not None:
        if episode not in parsed:
            available = ", ".join(sorted(parsed))
            raise KeyError(f"Episode {episode!r} not found for {task_key}; available episodes: {available}")
        selected_episode = episode
    else:
        current = np.asarray(current_joints, dtype=np.float64)
        if current.shape != (6,) or not np.all(np.isfinite(current)):
            raise ValueError("Current joint state must contain 6 finite values")
        # Minimize the largest single-joint move; the episode name breaks ties
        # deterministically.
        selected_episode = min(
            parsed,
            key=lambda name: (float(np.max(np.abs(parsed[name][:6] - current))), name),
        )

    selected = parsed[selected_episode]
    return TaskInitialState(
        task_key=task_key,
        episode=selected_episode,
        joints=selected[:6].copy(),
        gripper=float(selected[6]),
        robot_type=payload.get("robot_type"),
    )


def smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)
