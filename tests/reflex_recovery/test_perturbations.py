import jax
import jax.numpy as jnp
import numpy as np
import mujoco

from musclemimic.research.reflex_recovery.perturbations import (
    apply_generalized_torque,
    discrete_impulse_nms,
    torque_pulse,
    resolve_joint_dof_index,
    initialize_perturbation_state,
    reset_perturbation_state,
    triggered_torque,
    update_perturbation_trigger,
)


def test_rectangular_sign_duration_impulse_and_disabled():
    time = jnp.arange(8) * 0.01
    torque = jax.vmap(lambda t: torque_pulse(t, 0.02, 0.03, 10.0, -1, "rectangular"))(time)
    np.testing.assert_allclose(torque, [0, 0, -10, -10, -10, 0, 0, 0], atol=1e-5)
    np.testing.assert_allclose(discrete_impulse_nms(torque, 0.01), -0.3, atol=1e-6)
    disabled = torque_pulse(time, 0.02, 0.03, 10.0, 1, "rectangular", False)
    np.testing.assert_array_equal(disabled, np.zeros(8))


def test_generalized_torque_has_no_cross_environment_contamination():
    force = jnp.zeros((3, 5))
    output = apply_generalized_torque(force, jnp.array([0, 2, 4]), jnp.array([1.0, -2.0, 3.0]))
    expected = np.zeros((3, 5))
    expected[0, 0], expected[1, 2], expected[2, 4] = 1.0, -2.0, 3.0
    np.testing.assert_allclose(output, expected)


def test_supported_joint_names_resolve_to_exact_dofs():
    model = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody><body><joint name='hip_flexion_r'/><joint name='knee_angle_l'/><joint name='ankle_angle_r'/><geom type='sphere' size='.1'/></body></worldbody></mujoco>"""
    )
    assert resolve_joint_dof_index(model, "hip", "right") == 0
    assert resolve_joint_dof_index(model, "knee", "left") == 1
    assert resolve_joint_dof_index(model, "ankle", "right") == 2


def test_phase_trigger_fires_once_and_resets_per_environment():
    state = initialize_perturbation_state(2)
    np.testing.assert_array_equal(
        triggered_torque(state, jnp.array([0, 0]), 0.01, 0.03, 10.0, 1, "rectangular"), [0.0, 0.0]
    )
    update = jax.jit(
        lambda s, step, phase: update_perturbation_trigger(
            s, step, 0.01, "phase", phase_fraction=phase, onset_phase=0.25, phase_tolerance=0.02
        )
    )
    state, fire = update(state, jnp.array([10, 10]), jnp.array([0.25, 0.5]))
    np.testing.assert_array_equal(fire, [True, False])
    state, fire = update(state, jnp.array([11, 11]), jnp.array([0.25, 0.25]))
    np.testing.assert_array_equal(fire, [False, True])
    torque = triggered_torque(state, jnp.array([12, 12]), 0.01, 0.03, 10.0, 1, "rectangular")
    np.testing.assert_allclose(torque, [10.0, 10.0])
    state = reset_perturbation_state(state, jnp.array([True, False]))
    np.testing.assert_array_equal(state.triggered, [False, True])
    np.testing.assert_array_equal(state.onset_step, [-1, 11])
