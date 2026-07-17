import jax
import jax.numpy as jnp
import numpy as np

from musclemimic.research.reflex_recovery.metrics import classify_recovery


def test_recovery_requires_stable_dwell_and_no_nonfoot_contact():
    stable = jnp.array([[False, True, True, True], [False, True, False, True]])
    result = jax.jit(classify_recovery, static_argnames=("dwell_steps",))(
        jnp.zeros_like(stable),
        jnp.ones_like(stable, dtype=jnp.float32),
        jnp.zeros_like(stable, dtype=jnp.float32),
        stable,
        0.8,
        0.3,
        3,
    )
    np.testing.assert_array_equal(result.recovered, [True, False])
    np.testing.assert_array_equal(result.recovery_step, [3, -1])
