import jax
import jax.numpy as jnp
import numpy as np

from musclemimic.research.reflex_recovery.delay_buffer import initialize_delay_buffer, push_and_read, reset_environments


def test_delay_and_per_environment_reset_are_jittable():
    state = initialize_delay_buffer(2, 3, 1)
    step = jax.jit(lambda s, x: push_and_read(s, x, 2))
    outputs = []
    for value in (1.0, 2.0, 3.0):
        state, output = step(state, jnp.full((2, 1), value))
        outputs.append(np.asarray(output[:, 0]))
    np.testing.assert_allclose(outputs, [[0, 0], [0, 0], [1, 1]])
    state = reset_environments(state, jnp.array([True, False]))
    np.testing.assert_allclose(np.asarray(state.values[0]), 0.0)
    assert np.any(np.asarray(state.values[1]) != 0.0)
