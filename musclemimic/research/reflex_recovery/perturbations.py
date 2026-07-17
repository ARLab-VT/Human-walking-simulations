"""Vectorized torque-pulse generation independent of policy actions."""

from __future__ import annotations

import jax.numpy as jnp
import mujoco
from flax import struct


@struct.dataclass
class PerturbationState:
    """Per-environment one-shot trigger state."""

    triggered: jnp.ndarray
    onset_step: jnp.ndarray


def initialize_perturbation_state(num_envs: int | None = None) -> PerturbationState:
    """Create scalar or vector one-shot state."""
    shape = () if num_envs is None else (num_envs,)
    return PerturbationState(
        triggered=jnp.zeros(shape, dtype=bool),
        onset_step=jnp.full(shape, -1, dtype=jnp.int32),
    )


def reset_perturbation_state(state: PerturbationState, reset_mask: jnp.ndarray) -> PerturbationState:
    """Reset only selected vectorized environments."""
    return PerturbationState(
        triggered=jnp.where(reset_mask, False, state.triggered),
        onset_step=jnp.where(reset_mask, -1, state.onset_step),
    )


def update_perturbation_trigger(
    state: PerturbationState,
    step: jnp.ndarray,
    control_dt_s: float,
    onset_mode: str,
    onset_time_s: float | None = None,
    phase_fraction: jnp.ndarray | None = None,
    onset_phase: float | None = None,
    phase_tolerance: float = 0.02,
    heel_strike_event: jnp.ndarray | None = None,
    stance_fraction: jnp.ndarray | None = None,
) -> tuple[PerturbationState, jnp.ndarray]:
    """Update a one-shot time, phase, heel-strike, or stance trigger."""
    if onset_mode == "time":
        if onset_time_s is None:
            raise ValueError("onset_time_s is required")
        condition = step * control_dt_s >= onset_time_s
    elif onset_mode == "phase":
        if phase_fraction is None or onset_phase is None:
            raise ValueError("phase_fraction and onset_phase are required")
        circular_distance = jnp.abs(((phase_fraction - onset_phase + 0.5) % 1.0) - 0.5)
        condition = circular_distance <= phase_tolerance
    elif onset_mode == "heel_strike":
        if heel_strike_event is None:
            raise ValueError("heel_strike_event is required")
        condition = heel_strike_event
    elif onset_mode == "stance_percentage":
        if stance_fraction is None or onset_phase is None:
            raise ValueError("stance_fraction and onset_phase are required")
        condition = jnp.abs(stance_fraction - onset_phase) <= phase_tolerance
    else:
        raise ValueError(f"Unsupported onset mode: {onset_mode}")
    fire = condition & ~state.triggered
    return PerturbationState(
        triggered=state.triggered | fire,
        onset_step=jnp.where(fire, step, state.onset_step),
    ), fire


def triggered_torque(
    state: PerturbationState,
    step: jnp.ndarray,
    control_dt_s: float,
    duration_s: float,
    magnitude_nm: float,
    direction: int,
    waveform: str,
) -> jnp.ndarray:
    """Generate torque relative to each environment's recorded onset step."""
    progress = ((step - state.onset_step) * control_dt_s) / duration_s
    torque_nm = direction * magnitude_nm * waveform_value(progress, waveform)
    return jnp.where(state.triggered, torque_nm, 0.0)


def waveform_value(progress: jnp.ndarray, waveform: str, backend=jnp) -> jnp.ndarray:
    """Return unit-amplitude waveform for normalized progress in `[0, 1)`."""
    # A small boundary tolerance prevents float32 division from adding an
    # unintended sample at exactly onset + duration.
    active = (progress >= 0.0) & (progress < 1.0 - 1e-6)
    if waveform == "rectangular":
        value = backend.ones_like(progress)
    elif waveform == "half_sine":
        value = backend.sin(backend.pi * progress)
    elif waveform == "triangular":
        value = 1.0 - backend.abs((2.0 * progress) - 1.0)
    else:
        raise ValueError(f"Unsupported waveform: {waveform}")
    return backend.where(active, value, 0.0)


def torque_pulse(
    time_s: jnp.ndarray,
    onset_s: jnp.ndarray,
    duration_s: jnp.ndarray,
    magnitude_nm: jnp.ndarray,
    direction: jnp.ndarray,
    waveform: str,
    enabled: jnp.ndarray | bool = True,
    backend=jnp,
) -> jnp.ndarray:
    """Generate per-environment torque in Nm from vector inputs shaped `[env]`."""
    progress = (time_s - onset_s) / duration_s
    return backend.where(enabled, direction * magnitude_nm * waveform_value(progress, waveform, backend), 0.0)


def resolve_joint_dof_index(model: mujoco.MjModel, joint: str, side: str) -> int:
    """Resolve a supported anatomical joint and side to its generalized-force index."""
    joint_names = {
        "hip": "hip_flexion_{side}",
        "knee": "knee_angle_{side}",
        "ankle": "ankle_angle_{side}",
    }
    if joint not in joint_names:
        raise ValueError(f"Unsupported perturbation joint: {joint}")
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported perturbation side: {side}")
    suffix = "l" if side == "left" else "r"
    name = joint_names[joint].format(side=suffix)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"Joint {name!r} is missing from the compiled model")
    return int(model.jnt_dofadr[joint_id])


def discrete_impulse_nms(torque_nm: jnp.ndarray, simulation_dt_s: float) -> jnp.ndarray:
    """Integrate torque samples along the last axis into impulse in N m s."""
    return jnp.sum(torque_nm, axis=-1) * simulation_dt_s


def apply_generalized_torque(qfrc_applied: jnp.ndarray, dof_index: jnp.ndarray, torque_nm: jnp.ndarray) -> jnp.ndarray:
    """Add one independent joint torque per environment to `[env, nv]` forces."""
    rows = jnp.arange(qfrc_applied.shape[0])
    return qfrc_applied.at[rows, dof_index].add(torque_nm)
