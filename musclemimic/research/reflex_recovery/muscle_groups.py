"""Functional muscle-group validation and differentiable distribution."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class MuscleGroup:
    """A named functional group mapped to compiled actuator indices."""

    name: str
    actuator_names: tuple[str, ...]
    actuator_indices: tuple[int, ...]
    weights: tuple[float, ...]
    side: str
    function: str


def validate_groups(groups: tuple[MuscleGroup, ...], actuator_names: tuple[str, ...], allow_overlap: bool = False) -> None:
    """Validate names, indices, normalized weights, and accidental overlap."""
    seen: set[int] = set()
    for group in groups:
        if not (len(group.actuator_names) == len(group.actuator_indices) == len(group.weights)):
            raise ValueError(f"Mismatched mapping lengths for {group.name}")
        if not group.actuator_names:
            warnings.warn(f"Muscle group {group.name} is empty", stacklevel=2)
            continue
        for name, index in zip(group.actuator_names, group.actuator_indices, strict=True):
            if not 0 <= index < len(actuator_names) or actuator_names[index] != name:
                raise ValueError(f"Actuator mapping mismatch for {group.name}: {name}@{index}")
            if index in seen and not allow_overlap:
                raise ValueError(f"Actuator {name} belongs to multiple groups")
            seen.add(index)
        if not np.isclose(sum(abs(weight) for weight in group.weights), 1.0):
            raise ValueError(f"Absolute weights for {group.name} must sum to one")


def distribution_matrix(groups: tuple[MuscleGroup, ...], num_actuators: int) -> jnp.ndarray:
    """Build `[group, actuator]` matrix outside the JIT-compiled step."""
    matrix = np.zeros((len(groups), num_actuators), dtype=np.float32)
    for group_index, group in enumerate(groups):
        matrix[group_index, np.asarray(group.actuator_indices)] = np.asarray(group.weights)
    return jnp.asarray(matrix)


def distribute_group_action(group_action: jnp.ndarray, group_to_actuator: jnp.ndarray) -> jnp.ndarray:
    """Distribute `[env, group]` actions into `[env, actuator]` residuals."""
    return jnp.matmul(group_action, group_to_actuator)


_LOWER_BODY_FUNCTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hip_flexors", ("iliacus", "psoas", "sart")),
    ("hip_extensors", ("glmax1", "glmax2", "glmax3")),
    ("hip_abductors", ("glmed1", "glmed2", "glmed3", "glmin1", "glmin2", "glmin3", "piri", "tfl")),
    ("hip_adductors", ("addbrev", "addlong", "addmagDist", "addmagIsch", "addmagMid", "addmagProx", "grac")),
    ("knee_extensors", ("recfem", "vasint", "vaslat", "vasmed")),
    ("knee_flexors", ("bflh", "bfsh", "semimem", "semiten")),
    ("ankle_plantarflexors", ("gaslat", "gasmed", "soleus", "tibpost", "perbrev", "perlong", "fdl", "fhl")),
    ("ankle_dorsiflexors", ("tibant", "edl", "ehl")),
)


def build_lower_body_groups(actuator_names: tuple[str, ...]) -> tuple[MuscleGroup, ...]:
    """Build 16 exclusive bilateral groups from compiled actuator names.

    Equal positive weights sum to one within each group. Biarticular muscles
    are assigned once to avoid hidden double counting in the initial controller.
    """
    name_to_index = {name: index for index, name in enumerate(actuator_names)}
    groups: list[MuscleGroup] = []
    for side, suffix in (("right", "r"), ("left", "l")):
        for function, base_names in _LOWER_BODY_FUNCTIONS:
            expected = tuple(f"{name}_{suffix}" for name in base_names)
            present = tuple(name for name in expected if name in name_to_index)
            missing = tuple(name for name in expected if name not in name_to_index)
            if missing:
                warnings.warn(f"Missing expected actuators for {side}_{function}: {missing}", stacklevel=2)
            if not present:
                raise ValueError(f"No actuators found for {side}_{function}")
            weight = 1.0 / len(present)
            groups.append(
                MuscleGroup(
                    name=f"{side}_{function}",
                    actuator_names=present,
                    actuator_indices=tuple(name_to_index[name] for name in present),
                    weights=(weight,) * len(present),
                    side=side,
                    function=function,
                )
            )
    result = tuple(groups)
    validate_groups(result, actuator_names)
    return result


def export_group_map(groups: tuple[MuscleGroup, ...], output_dir: str | Path) -> None:
    """Export group membership to tidy CSV and readable Markdown."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "group": group.name,
            "side": group.side,
            "function": group.function,
            "actuator_name": name,
            "actuator_index": index,
            "weight": weight,
        }
        for group in groups
        for name, index, weight in zip(group.actuator_names, group.actuator_indices, group.weights, strict=True)
    ]
    with (output / "muscle_groups.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Functional Muscle Groups", "", "Equal weights are normalized within each exclusive group.", ""]
    for group in groups:
        lines.extend((f"## {group.name}", "", ", ".join(group.actuator_names), ""))
    (output / "muscle_groups.md").write_text("\n".join(lines), encoding="utf-8")
