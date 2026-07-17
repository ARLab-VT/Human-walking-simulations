"""Small grouped-action recovery policy."""

from __future__ import annotations

from collections.abc import Sequence

from flax import linen as nn
import jax.numpy as jnp


class RecoveryPolicy(nn.Module):
    """Bounded MLP mapping compact recovery observations to group actions."""

    num_group_actions: int
    hidden_layers: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Map observations `[..., observation]` to `[..., group]` in `[-1, 1]`."""
        features = observation
        for width in self.hidden_layers:
            features = nn.silu(nn.Dense(width)(features))
        return jnp.tanh(nn.Dense(self.num_group_actions)(features))


def count_parameters(parameters: dict) -> int:
    """Count scalar leaves in a Flax parameter tree."""
    from jax import tree_util

    return sum(leaf.size for leaf in tree_util.tree_leaves(parameters))
