import jax
import jax.numpy as jnp
import numpy as np

from musclemimic.research.reflex_recovery.action_composer import compose_action


def test_composition_gate_limits_and_diagnostics():
    compose = jax.jit(compose_action)
    result = compose(
        jnp.array([[0.9, 0.0], [0.0, 0.0]]),
        jnp.array([[0.5, 0.0], [0.0, 0.0]]),
        jnp.ones((2, 2)),
        jnp.array([1.0, 0.0]),
        jnp.zeros((2, 2)),
        0.01,
        rate_limit_per_s=1000.0,
    )
    np.testing.assert_allclose(result.action, [[1.0, 0.5], [0.0, 0.0]])
    np.testing.assert_allclose(result.saturation_fraction, [0.5, 0.0])
    np.testing.assert_allclose(result.recovery_residual[1], [0.0, 0.0])


def test_rate_limit_is_applied():
    result = compose_action(jnp.ones((1, 2)), jnp.zeros((1, 2)), jnp.zeros((1, 2)), jnp.zeros(1), jnp.zeros((1, 2)), 0.01, rate_limit_per_s=2.0)
    np.testing.assert_allclose(result.action, [[0.02, 0.02]], atol=1e-7)
    np.testing.assert_allclose(result.rate_violation_fraction, [1.0])
