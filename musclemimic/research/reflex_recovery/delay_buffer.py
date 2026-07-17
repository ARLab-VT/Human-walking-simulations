"""JAX-compatible per-environment sensory delay ring buffer."""

from __future__ import annotations

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class DelayBufferState:
    """Ring-buffer values `[env, history, feature]` and scalar write index."""

    values: jnp.ndarray
    write_index: jnp.ndarray


def delay_steps(delay_s: float, control_dt_s: float) -> int:
    """Convert seconds to nearest control steps, documenting nearest-step rounding."""
    if delay_s < 0 or control_dt_s <= 0:
        raise ValueError("delay_s must be nonnegative and control_dt_s positive")
    return int(round(delay_s / control_dt_s))


def initialize_delay_buffer(num_envs: int, history_steps: int, num_features: int) -> DelayBufferState:
    """Create a zero-filled ring buffer with at least one history slot."""
    if min(num_envs, history_steps, num_features) <= 0:
        raise ValueError("all buffer dimensions must be positive")
    return DelayBufferState(
        values=jnp.zeros((num_envs, history_steps, num_features)),
        write_index=jnp.asarray(0, dtype=jnp.int32),
    )


def push_and_read(
    state: DelayBufferState, signals: jnp.ndarray, read_delay_steps: int
) -> tuple[DelayBufferState, jnp.ndarray]:
    """Push `[env, feature]` signals and read the requested delayed sample."""
    history = state.values.shape[1]
    if not 0 <= read_delay_steps < history:
        raise ValueError("read_delay_steps must be within buffer history")
    values = state.values.at[:, state.write_index, :].set(signals)
    read_index = (state.write_index - read_delay_steps) % history
    output = values[:, read_index, :]
    return state.replace(values=values, write_index=(state.write_index + 1) % history), output


def reset_environments(state: DelayBufferState, reset_mask: jnp.ndarray) -> DelayBufferState:
    """Clear history only for environments where `reset_mask[env]` is true."""
    values = jnp.where(reset_mask[:, None, None], jnp.zeros_like(state.values), state.values)
    return state.replace(values=values)
