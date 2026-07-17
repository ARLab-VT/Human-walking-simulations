"""Frozen-call wrapper for a restored MuscleMimic policy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax


class FrozenPolicyOutput(NamedTuple):
    """Base action plus explicitly stopped parameter tree."""

    action: jax.Array
    frozen_parameters: Any


def call_frozen_base_policy(
    apply_function: Callable[..., jax.Array], parameters: Any, observation: jax.Array, *args: Any, **kwargs: Any
) -> FrozenPolicyOutput:
    """Call the base policy with stop-gradient applied to every parameter leaf."""
    frozen = jax.tree.map(jax.lax.stop_gradient, parameters)
    return FrozenPolicyOutput(apply_function(frozen, observation, *args, **kwargs), frozen)
