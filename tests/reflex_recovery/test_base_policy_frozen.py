import jax
import jax.numpy as jnp
import numpy as np
import optax

from musclemimic.research.reflex_recovery.base_policy import call_frozen_base_policy
from musclemimic.research.reflex_recovery.recovery_policy import RecoveryPolicy


def test_base_parameters_are_bitwise_unchanged_during_recovery_update():
    base = {"w": jnp.array([[2.0]])}
    policy = RecoveryPolicy(num_group_actions=1, hidden_layers=(4,))
    recovery = policy.init(jax.random.key(0), jnp.ones((1, 2)))
    optimizer = optax.sgd(0.1)
    optimizer_state = optimizer.init(recovery)

    def loss(params):
        base_action = call_frozen_base_policy(lambda p, x: x @ p["w"], base, jnp.ones((1, 1))).action
        return jnp.sum((base_action + policy.apply(params, jnp.ones((1, 2)))) ** 2)

    gradients = jax.grad(loss)(recovery)
    updates, optimizer_state = optimizer.update(gradients, optimizer_state)
    recovery = optax.apply_updates(recovery, updates)
    np.testing.assert_array_equal(base["w"], np.array([[2.0]]))
    assert any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(recovery))
