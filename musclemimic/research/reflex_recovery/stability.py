"""Normalized robust stability-risk components."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class StabilityRisk(NamedTuple):
    """Each component and total has shape `[env]`."""

    tilt: jnp.ndarray
    angular_velocity: jnp.ndarray
    height: jnp.ndarray
    contact: jnp.ndarray
    reference_deviation: jnp.ndarray
    total: jnp.ndarray


def positive_excess(value: jnp.ndarray, threshold: float, scale: float) -> jnp.ndarray:
    """Continuous nonnegative threshold exceedance."""
    return jnp.maximum(jnp.abs(value) - threshold, 0.0) / scale


def compute_stability_risk(
    pelvis_roll_pitch_rad: jnp.ndarray,
    pelvis_angular_velocity_rad_s: jnp.ndarray,
    pelvis_height_m: jnp.ndarray,
    foot_contacts: jnp.ndarray,
    reference_deviation_m: jnp.ndarray,
    weights: jnp.ndarray,
) -> StabilityRisk:
    """Compute robust risk from `[env, feature]` state arrays."""
    tilt = jnp.max(positive_excess(pelvis_roll_pitch_rad, 0.15, 0.35), axis=-1)
    angular_velocity = jnp.max(positive_excess(pelvis_angular_velocity_rad_s, 0.5, 2.0), axis=-1)
    height = jnp.maximum(0.8 - pelvis_height_m, 0.0) / 0.3
    contact = jnp.where(jnp.any(foot_contacts > 0.5, axis=-1), 0.0, 1.0)
    reference_deviation = jnp.maximum(reference_deviation_m - 0.1, 0.0) / 0.5
    components = jnp.stack((tilt, angular_velocity, height, contact, reference_deviation), axis=-1)
    return StabilityRisk(tilt, angular_velocity, height, contact, reference_deviation, components @ weights)
