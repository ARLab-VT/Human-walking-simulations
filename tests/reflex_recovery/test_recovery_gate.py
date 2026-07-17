import jax
import jax.numpy as jnp

from musclemimic.research.reflex_recovery.recovery_gate import update_recovery_gate


def test_gate_is_bounded_monotonic_and_jittable():
    update = jax.jit(update_recovery_gate)
    previous = jnp.zeros(3)
    beta = update(jnp.array([0.0, 1.0, 2.0]), previous)
    assert 0 <= float(beta[0]) < float(beta[1]) < float(beta[2]) <= 1
    beta2 = update(jnp.array([0.0, 1.0, 2.0]), beta)
    assert float(beta2[2]) > float(beta[2])
