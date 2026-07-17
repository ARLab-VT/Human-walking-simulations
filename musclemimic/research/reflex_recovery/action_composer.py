"""Safe composition in normalized MuscleMimic policy-action space."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class ActionComposition(NamedTuple):
    """Composed action and per-environment safety diagnostics."""

    action: jnp.ndarray
    unclipped_action: jnp.ndarray
    reflex_residual: jnp.ndarray
    recovery_residual: jnp.ndarray
    saturation_fraction: jnp.ndarray
    rate_violation_fraction: jnp.ndarray


def compose_action(
    base_action: jnp.ndarray,
    reflex_action: jnp.ndarray,
    recovery_action: jnp.ndarray,
    beta: jnp.ndarray,
    previous_action: jnp.ndarray,
    control_dt_s: float,
    reflex_scale: float = 1.0,
    recovery_scale: float = 1.0,
    residual_limit: float = 0.5,
    rate_limit_per_s: float = 10.0,
    action_low: float | jnp.ndarray = -1.0,
    action_high: float | jnp.ndarray = 1.0,
) -> ActionComposition:
    """Compose `[env, actuator]` components and enforce magnitude/rate/actuator limits."""
    reflex = jnp.clip(jnp.nan_to_num(reflex_action), -residual_limit, residual_limit) * reflex_scale
    recovery = jnp.clip(jnp.nan_to_num(recovery_action), -residual_limit, residual_limit)
    recovery = recovery * recovery_scale * beta[..., None]
    unclipped = jnp.nan_to_num(base_action) + reflex + recovery
    magnitude_clipped = jnp.clip(unclipped, action_low, action_high)
    max_delta = rate_limit_per_s * control_dt_s
    delta = magnitude_clipped - previous_action
    action = previous_action + jnp.clip(delta, -max_delta, max_delta)
    action = jnp.clip(action, action_low, action_high)
    saturation = jnp.mean((unclipped < action_low) | (unclipped > action_high), axis=-1)
    rate_violation = jnp.mean(jnp.abs(delta) > max_delta, axis=-1)
    return ActionComposition(action, unclipped, reflex, recovery, saturation, rate_violation)
