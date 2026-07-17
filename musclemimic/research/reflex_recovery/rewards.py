"""Recovery-priority reward blending around the upstream nominal reward."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class RewardBlend(NamedTuple):
    """Blended reward and effective tracking weights."""

    reward: jnp.ndarray
    nominal_weight: jnp.ndarray
    recovery_weight: jnp.ndarray
    effective_imitation_weight: jnp.ndarray


def blend_rewards(
    nominal_reward: jnp.ndarray,
    recovery_reward: jnp.ndarray,
    beta: jnp.ndarray,
    imitation_weight: float,
    imitation_relaxation: float = 1.0,
    minimum_imitation_weight: float = 0.0,
) -> RewardBlend:
    """Blend per-environment rewards continuously without replacing nominal components."""
    clipped_beta = jnp.clip(beta, 0.0, 1.0)
    effective = imitation_weight * (1.0 - imitation_relaxation * clipped_beta)
    effective = jnp.maximum(effective, minimum_imitation_weight)
    return RewardBlend(
        (1.0 - clipped_beta) * nominal_reward + clipped_beta * recovery_reward,
        1.0 - clipped_beta,
        clipped_beta,
        effective,
    )


def recovery_reward(
    alive: jnp.ndarray,
    upright: jnp.ndarray,
    stable_contact: jnp.ndarray,
    angular_velocity_cost: jnp.ndarray,
    residual_cost: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Combine `[env]` recovery terms using five configured weights."""
    terms = jnp.stack((alive, upright, stable_contact, -angular_velocity_cost, -residual_cost), axis=-1)
    return terms @ weights
