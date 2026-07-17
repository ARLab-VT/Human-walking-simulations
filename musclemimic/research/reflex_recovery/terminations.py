"""Recovery-aware termination decisions with exact reason codes."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


TERMINATION_NONE = 0
TERMINATION_NON_FOOT_CONTACT = 1
TERMINATION_LOW_PELVIS = 2
TERMINATION_ORIENTATION = 3
TERMINATION_INVALID_STATE = 4
TERMINATION_REFERENCE = 5
TERMINATION_TIMEOUT = 6


class TerminationResult(NamedTuple):
    """Per-environment terminal mask and integer reason."""

    done: jnp.ndarray
    reason: jnp.ndarray


def recovery_aware_termination(
    non_foot_contact: jnp.ndarray,
    pelvis_height_m: jnp.ndarray,
    pelvis_tilt_rad: jnp.ndarray,
    valid_state: jnp.ndarray,
    reference_deviation_m: jnp.ndarray,
    timed_out: jnp.ndarray,
    beta: jnp.ndarray,
    minimum_pelvis_height_m: float = 0.45,
    maximum_tilt_rad: float = 1.3,
    maximum_reference_deviation_m: float = 1.0,
    reference_relaxation: float = 2.0,
) -> TerminationResult:
    """Retain true falls while relaxing only reference deviation during recovery."""
    reference_limit = maximum_reference_deviation_m * (1.0 + reference_relaxation * beta)
    conditions = (
        non_foot_contact,
        pelvis_height_m < minimum_pelvis_height_m,
        pelvis_tilt_rad > maximum_tilt_rad,
        ~valid_state,
        reference_deviation_m > reference_limit,
        timed_out,
    )
    done = jnp.zeros_like(non_foot_contact, dtype=bool)
    reason = jnp.zeros_like(pelvis_height_m, dtype=jnp.int32)
    for code, condition in enumerate(conditions, start=1):
        select = condition & ~done
        reason = jnp.where(select, code, reason)
        done = done | condition
    return TerminationResult(done, reason)
