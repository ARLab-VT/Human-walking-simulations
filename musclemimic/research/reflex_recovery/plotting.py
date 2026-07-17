"""Small plotting helpers for reflex-recovery notebooks and reports."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .rollout import BaselineRollout


def plot_baseline_summary(rollout: BaselineRollout):
    """Plot reward, policy-action envelope, and bilateral touch signals."""
    time = rollout.timesteps_s
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes[0].plot(time, rollout.rewards)
    axes[0].set_ylabel("reward")
    axes[1].fill_between(
        time,
        np.min(rollout.policy_actions, axis=-1),
        np.max(rollout.policy_actions, axis=-1),
        alpha=0.35,
    )
    axes[1].set_ylabel("policy action")
    if rollout.touch_observations.size:
        axes[2].plot(time, rollout.touch_observations)
    axes[2].set_ylabel("touch")
    axes[2].set_xlabel("time [s]")
    return figure, axes


def plot_perturbation_response(time_s, torque_nm, pelvis_risk=None):
    """Plot the independent perturbation torque and optional stability risk."""
    rows = 2 if pelvis_risk is not None else 1
    figure, axes = plt.subplots(rows, 1, figsize=(10, 3 * rows), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    axes[0].plot(time_s, torque_nm)
    axes[0].set_ylabel("torque [Nm]")
    if pelvis_risk is not None:
        axes[1].plot(time_s, pelvis_risk)
        axes[1].set_ylabel("stability risk")
    axes[-1].set_xlabel("time [s]")
    return figure, axes
