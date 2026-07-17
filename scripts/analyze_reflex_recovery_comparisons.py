#!/usr/bin/env python3
"""Create tidy metrics and publication-style plots for matched smoke rollouts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from musclemimic.research.reflex_recovery.comparison import (
    comparison_metrics,
    load_comparison_rollout,
    pelvis_tilt_rad,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/reflex_recovery/comparisons"))
    args = parser.parse_args()
    conditions = ("base_only", "base_perturbed", "base_plus_reflex")
    rollouts = {name: load_comparison_rollout(args.root, name) for name in conditions}
    metrics = [comparison_metrics(rollouts[name], rollouts["base_only"]) for name in conditions]
    with (args.root / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    figure, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
    colors = {"base_only": "black", "base_perturbed": "tab:red", "base_plus_reflex": "tab:blue"}
    for name, rollout in rollouts.items():
        axes[0].plot(rollout.time_s, rollout.qpos[:, 2], label=name, color=colors[name])
        axes[1].plot(rollout.time_s, pelvis_tilt_rad(rollout.qpos), label=name, color=colors[name])
        axes[2].plot(rollout.time_s, rollout.rewards, label=name, color=colors[name], alpha=0.85)
        if rollout.reflex_group_actions.size:
            axes[3].plot(
                rollout.time_s,
                np.max(np.abs(rollout.reflex_group_actions), axis=-1),
                label=name,
                color=colors[name],
            )
    torque = rollouts["base_perturbed"].perturbation_torque_nm
    active = np.flatnonzero(np.abs(torque) > 1e-9)
    if active.size:
        for axis in axes:
            axis.axvspan(
                rollouts["base_perturbed"].time_s[active[0]],
                rollouts["base_perturbed"].time_s[active[-1]],
                color="tab:orange",
                alpha=0.18,
            )
    axes[0].set_ylabel("pelvis height [m]")
    axes[1].set_ylabel("pelvis tilt [rad]")
    axes[2].set_ylabel("reward")
    axes[3].set_ylabel("max |reflex group|")
    axes[3].set_xlabel("time [s]")
    axes[0].legend(ncol=3)
    figure.savefig(args.root / "matched_rollout_comparison.png", dpi=180)
    print(args.root / "summary.csv")
    print(args.root / "matched_rollout_comparison.png")


if __name__ == "__main__":
    main()
