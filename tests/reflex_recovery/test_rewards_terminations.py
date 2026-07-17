import jax
import jax.numpy as jnp
import numpy as np

from musclemimic.research.reflex_recovery.rewards import blend_rewards
from musclemimic.research.reflex_recovery.terminations import TERMINATION_REFERENCE, recovery_aware_termination


def test_reward_blend_and_imitation_relaxation():
    result = jax.jit(blend_rewards)(jnp.array([10.0, 10.0]), jnp.array([2.0, 2.0]), jnp.array([0.0, 1.0]), 1.0, 0.8, 0.1)
    np.testing.assert_allclose(result.reward, [10.0, 2.0])
    np.testing.assert_allclose(result.effective_imitation_weight, [1.0, 0.2])


def test_reference_termination_is_relaxed_but_fall_is_not():
    result = recovery_aware_termination(
        jnp.array([False, False, True]),
        jnp.ones(3),
        jnp.zeros(3),
        jnp.ones(3, dtype=bool),
        jnp.array([1.5, 1.5, 0.0]),
        jnp.zeros(3, dtype=bool),
        jnp.array([0.0, 1.0, 1.0]),
    )
    np.testing.assert_array_equal(result.done, [True, False, True])
    assert int(result.reason[0]) == TERMINATION_REFERENCE
    assert int(result.reason[2]) == 1
