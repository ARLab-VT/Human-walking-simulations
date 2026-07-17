"""Typed access to deterministic MuscleMimic rollout exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BaselineRollout:
    """One exported episode with time-major state and action arrays."""

    trajectory_id: int
    trajectory_qpos: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_accelerations: np.ndarray
    touch_observations: np.ndarray
    policy_actions: np.ndarray
    muscle_commands: np.ndarray
    muscle_activations: np.ndarray
    rewards: np.ndarray
    timesteps_s: np.ndarray
    joint_names: tuple[str, ...]
    control_dt_s: float
    environment_name: str
    backend: str

    @property
    def num_steps(self) -> int:
        """Number of saved control steps."""
        return int(self.rewards.shape[0])

    @property
    def total_reward(self) -> float:
        """Sum of per-step upstream rewards."""
        return float(np.sum(self.rewards))


def load_baseline_rollout(path: str | Path, episode: int = 0) -> BaselineRollout:
    """Load one episode from an upstream `--export_trajectory` NPZ file."""
    prefix = f"episode_{episode}_"
    with np.load(path, allow_pickle=False) as data:
        if episode < 0 or episode >= int(data["n_episodes"]):
            raise IndexError(f"Episode {episode} is not present in {path}")
        return BaselineRollout(
            trajectory_id=int(data[prefix + "traj_id"]),
            trajectory_qpos=data[prefix + "traj_qpos"].copy(),
            joint_positions=data[prefix + "joint_positions"].copy(),
            joint_velocities=data[prefix + "joint_velocities"].copy(),
            joint_accelerations=data[prefix + "joint_accelerations"].copy(),
            touch_observations=data[prefix + "touch_observations"].copy(),
            policy_actions=data[prefix + "policy_actions"].copy(),
            muscle_commands=data[prefix + "muscle_commands"].copy(),
            muscle_activations=data[prefix + "muscle_activations"].copy(),
            rewards=data[prefix + "rewards"].copy(),
            timesteps_s=data[prefix + "timesteps"].copy(),
            joint_names=tuple(str(name) for name in data["joint_names"]),
            control_dt_s=float(data["dt"]),
            environment_name=str(data["env_name"]),
            backend=str(data["backend"]),
        )


def scientific_arrays_equal(first: BaselineRollout, second: BaselineRollout) -> bool:
    """Compare deterministic scientific outputs while excluding wall-clock metadata."""
    arrays = (
        "trajectory_qpos",
        "joint_positions",
        "joint_velocities",
        "joint_accelerations",
        "touch_observations",
        "policy_actions",
        "muscle_commands",
        "muscle_activations",
        "rewards",
        "timesteps_s",
    )
    return (
        first.trajectory_id == second.trajectory_id
        and first.joint_names == second.joint_names
        and first.control_dt_s == second.control_dt_s
        and all(np.array_equal(getattr(first, name), getattr(second, name)) for name in arrays)
    )
