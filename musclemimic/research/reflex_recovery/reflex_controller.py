"""Interpretable grouped spinal-feedback-inspired residual controller."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class ReflexGains(NamedTuple):
    """Gain arrays with one value per functional muscle group."""

    position: jnp.ndarray
    velocity: jnp.ndarray
    load: jnp.ndarray
    pelvis_angle: jnp.ndarray
    pelvis_angular_velocity: jnp.ndarray
    bias: jnp.ndarray


def compute_reflex_action(
    joint_error: jnp.ndarray,
    joint_velocity: jnp.ndarray,
    limb_load: jnp.ndarray,
    pelvis_angle: jnp.ndarray,
    pelvis_angular_velocity: jnp.ndarray,
    stance_weight: jnp.ndarray,
    stance_gains: ReflexGains,
    swing_gains: ReflexGains,
    group_limit: float,
    enabled: bool = True,
) -> jnp.ndarray:
    """Compute saturated grouped output; feature arrays have shape `[env, group]`."""
    blend = stance_weight[..., None] if stance_weight.ndim == joint_error.ndim - 1 else stance_weight
    gains = ReflexGains(*(blend * stance + (1.0 - blend) * swing for stance, swing in zip(stance_gains, swing_gains, strict=True)))
    raw = (
        gains.position * joint_error
        + gains.velocity * joint_velocity
        + gains.load * limb_load
        + gains.pelvis_angle * pelvis_angle
        + gains.pelvis_angular_velocity * pelvis_angular_velocity
        + gains.bias
    )
    safe = jnp.nan_to_num(raw, nan=0.0, posinf=group_limit, neginf=-group_limit)
    return jnp.where(enabled, jnp.clip(safe, -group_limit, group_limit), jnp.zeros_like(safe))
