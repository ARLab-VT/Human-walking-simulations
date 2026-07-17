import jax
import jax.numpy as jnp
import numpy as np

from musclemimic.research.reflex_recovery.reflex_controller import ReflexGains, compute_reflex_action


def _gains(value):
    array = jnp.full((2,), value)
    zero = jnp.zeros((2,))
    return ReflexGains(array, zero, zero, zero, zero, zero)


def test_zero_gains_and_disabled_are_exactly_zero():
    features = jnp.ones((3, 2))
    zeros = _gains(0.0)
    function = jax.jit(compute_reflex_action, static_argnames=("enabled",))
    output = function(features, features, features, features, features, jnp.ones(3), zeros, zeros, 0.2, True)
    np.testing.assert_array_equal(output, np.zeros((3, 2)))
    output = function(features, features, features, features, features, jnp.ones(3), _gains(1.0), _gains(1.0), 0.2, False)
    np.testing.assert_array_equal(output, np.zeros((3, 2)))


def test_phase_blend_and_saturation():
    features = jnp.ones((2, 2))
    zeros = jnp.zeros_like(features)
    output = compute_reflex_action(features, zeros, zeros, zeros, zeros, jnp.array([1.0, 0.0]), _gains(2.0), _gains(-1.0), 0.5)
    np.testing.assert_allclose(output, [[0.5, 0.5], [-0.5, -0.5]])


def test_per_group_stance_blending():
    zeros = jnp.zeros(4)
    stance = ReflexGains(zeros, zeros, zeros, zeros, zeros, jnp.ones(4))
    swing = ReflexGains(zeros, zeros, zeros, zeros, zeros, -jnp.ones(4))
    output = compute_reflex_action(
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        jnp.asarray([1.0, 0.0, 0.25, 0.75]),
        stance,
        swing,
        group_limit=2.0,
    )
    np.testing.assert_allclose(output, [1.0, -1.0, -0.5, 0.5])
