"""Matched-condition rollout loading and descriptive comparison metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ComparisonRollout:
    """Arrays exported for one deterministic reflex-recovery condition."""

    condition: str
    path: Path
    time_s: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    policy_actions: np.ndarray
    muscle_commands: np.ndarray
    rewards: np.ndarray
    touch: np.ndarray
    perturbation_torque_nm: np.ndarray
    reflex_group_actions: np.ndarray
    reflex_saturation_fraction: np.ndarray
    done: np.ndarray
    control_dt_s: float

    @property
    def total_reward(self) -> float:
        return float(np.sum(self.rewards))

    @property
    def perturbation_onset_index(self) -> int:
        active = np.flatnonzero(np.abs(self.perturbation_torque_nm) > 1e-9)
        return int(active[0]) if active.size else -1


def load_comparison_rollout(root: str | Path, condition: str) -> ComparisonRollout:
    """Load the newest exported episode for a named comparison condition."""
    paths = sorted((Path(root) / condition).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No exported rollout for {condition!r} under {root}")
    path = paths[-1]
    prefix = "episode_0_"
    with np.load(path, allow_pickle=False) as data:
        return ComparisonRollout(
            condition=condition,
            path=path,
            time_s=data[prefix + "timesteps"].copy(),
            qpos=data[prefix + "joint_positions"].copy(),
            qvel=data[prefix + "joint_velocities"].copy(),
            policy_actions=data[prefix + "policy_actions"].copy(),
            muscle_commands=data[prefix + "muscle_commands"].copy(),
            rewards=data[prefix + "rewards"].copy(),
            touch=data[prefix + "touch_observations"].copy(),
            perturbation_torque_nm=data[prefix + "perturbation_torque_nm"].copy(),
            reflex_group_actions=data[prefix + "reflex_group_actions"].copy(),
            reflex_saturation_fraction=data[prefix + "reflex_saturation_fraction"].copy(),
            done=data[prefix + "done"].copy(),
            control_dt_s=float(data["dt"]),
        )


def pelvis_tilt_rad(qpos: np.ndarray) -> np.ndarray:
    """Return root tilt angle from MuJoCo free-joint quaternion `[w,x,y,z]`."""
    quaternion = np.asarray(qpos)[:, 3:7]
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    quaternion = quaternion / np.maximum(norm, 1e-12)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return np.arccos(np.clip(up_z, -1.0, 1.0))


def comparison_metrics(rollout: ComparisonRollout, nominal: ComparisonRollout) -> dict[str, float | str]:
    """Compute descriptive matched-rollout metrics; no inferential claim is implied."""
    length = min(len(rollout.time_s), len(nominal.time_s))
    onset = rollout.perturbation_onset_index
    comparison_start = onset if onset >= 0 else 0
    delta = rollout.qpos[:length] - nominal.qpos[:length]
    post_delta = delta[comparison_start:length]
    contact = np.any(rollout.touch[:length] > 0.0, axis=-1) if rollout.touch.shape[-1] else np.ones(length, bool)
    reflex = rollout.reflex_group_actions
    return {
        "condition": rollout.condition,
        "steps": float(length),
        "total_reward": rollout.total_reward,
        "reward_delta_vs_nominal": rollout.total_reward - nominal.total_reward,
        "minimum_pelvis_height_m": float(np.min(rollout.qpos[:length, 2])),
        "peak_pelvis_tilt_rad": float(np.max(pelvis_tilt_rad(rollout.qpos[:length]))),
        "post_onset_qpos_rmse_vs_nominal": float(np.sqrt(np.mean(post_delta * post_delta))),
        "contact_fraction": float(np.mean(contact)),
        "torque_impulse_nms": float(np.sum(rollout.perturbation_torque_nm) * rollout.control_dt_s),
        "peak_reflex_group_action": float(np.max(np.abs(reflex))) if reflex.size else 0.0,
        "mean_reflex_saturation_fraction": float(np.mean(rollout.reflex_saturation_fraction)),
        "done_fraction": float(np.mean(rollout.done)),
    }
