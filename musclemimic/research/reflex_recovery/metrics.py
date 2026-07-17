"""Scientific recovery classification and rollout metrics."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class RecoveryClassification(NamedTuple):
    """Per-environment success mask and first stable sample index."""

    recovered: jnp.ndarray
    recovery_step: jnp.ndarray


def classify_recovery(
    non_foot_contact: jnp.ndarray,
    pelvis_height_m: jnp.ndarray,
    pelvis_tilt_rad: jnp.ndarray,
    stable_support: jnp.ndarray,
    minimum_height_m: float,
    maximum_tilt_rad: float,
    dwell_steps: int,
) -> RecoveryClassification:
    """Classify `[env,time]` rollouts; non-termination alone is insufficient."""
    stable = (
        ~non_foot_contact
        & (pelvis_height_m >= minimum_height_m)
        & (pelvis_tilt_rad <= maximum_tilt_rad)
        & stable_support
    )
    window = jnp.ones((1, 1, dwell_steps), dtype=jnp.int32)
    counts = jnp.squeeze(
        jax.lax.conv_general_dilated(stable[:, None, :].astype(jnp.int32), window, (1,), "VALID"), axis=1
    )
    dwell = counts == dwell_steps
    recovered = jnp.any(dwell, axis=-1)
    first = jnp.argmax(dwell, axis=-1) + dwell_steps - 1
    return RecoveryClassification(recovered, jnp.where(recovered, first, -1))
