import jax.numpy as jnp
import numpy as np

from musclemimic.research.reflex_recovery.stability import compute_stability_risk


def test_unstable_state_has_higher_component_risks():
    risk = compute_stability_risk(
        jnp.array([[0.0, 0.0], [0.8, 0.0]]),
        jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        jnp.array([1.0, 0.5]),
        jnp.array([[1.0, 0.0], [0.0, 0.0]]),
        jnp.array([0.0, 0.8]),
        jnp.ones(5),
    )
    assert float(risk.total[1]) > float(risk.total[0])
    np.testing.assert_allclose(risk.total[0], 0.0)
