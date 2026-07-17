"""Continuous low-pass recovery activation."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def update_recovery_gate(
    risk: jnp.ndarray,
    previous_beta: jnp.ndarray,
    risk_threshold: float = 1.0,
    sigmoid_gain: float = 6.0,
    low_pass_fraction: float = 0.2,
    off_hysteresis: float = 0.1,
) -> jnp.ndarray:
    """Map risk to beta `[0,1]` with state-dependent hysteresis and smoothing."""
    threshold = risk_threshold - jnp.where(previous_beta > 0.5, off_hysteresis, 0.0)
    target = jax.nn.sigmoid(sigmoid_gain * (risk - threshold))
    beta = previous_beta + low_pass_fraction * (target - previous_beta)
    return jnp.clip(beta, 0.0, 1.0)
