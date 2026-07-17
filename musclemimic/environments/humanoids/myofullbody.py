"""
MyoFullBody environment - A full-body muscle-actuated humanoid model.
"""

import loco_mujoco
import mujoco
import jax.numpy as jnp
import numpy as np
from loco_mujoco.core import ObservationType
from loco_mujoco.core.utils import info_property
from musclemimic.environments import LocoEnv
from musclemimic.utils.logging import setup_logger
from mujoco import MjSpec
import musclemimic_models
from musclemimic.research.reflex_recovery.config import PerturbationConfig
from musclemimic.research.reflex_recovery.muscle_groups import (
    build_lower_body_groups,
    distribute_group_action,
    distribution_matrix,
)
from musclemimic.research.reflex_recovery.reflex_controller import ReflexGains, compute_reflex_action
from musclemimic.research.reflex_recovery.perturbations import (
    PerturbationState,
    resolve_joint_dof_index,
    torque_pulse,
    triggered_torque,
    update_perturbation_trigger,
)

logger = setup_logger(__name__, identifier="[MyoFullBody]")


class MyoFullBody(LocoEnv):
    """
    Description
    ------------

    MuJoCo environment of a full-body humanoid model with muscle actuation.
    This model combines the full skeletal structure with comprehensive muscle control
    throughout the entire body, including legs, torso, arms, and optional fingers.

    The model uses muscle actuators (MuJoCo type 4) with Hill-type muscle dynamics,
    providing biomechanically realistic force generation and movement patterns.

    .. note:: Control range for all muscles is modified from default [0,1] to [-1,1]
              to match the MyoBimanualArm convention.

    Default Observation Space
    -----------------

    The observation space consists of joint positions and velocities for the full body,
    plus optional muscle state observations (length, velocity, force, activation).

    Default Action Space
    -------------------

    Control function type: **DefaultControl**

    All muscle actuators use control range [-1.0, 1.0] where:
    - Negative values (-1.0 to 0.0) represent muscle relaxation to baseline activation
    - Positive values (0.0 to 1.0) represent increasing muscle activation levels

    Methods
    ------------

    """

    mjx_enabled = False

    def __init__(
        self,
        timestep: float = 0.002,
        n_substeps: int = 5,
        disable_fingers: bool = True,
        enable_joint_pos_observations: bool = True,
        enable_joint_vel_observations: bool = True,
        enable_muscle_length_observations: bool = False,
        enable_muscle_velocity_observations: bool = False,
        enable_muscle_force_observations: bool = False,
        enable_muscle_excitation_observations: bool = False,
        enable_muscle_activation_observations: bool = False,
        enable_touch_sensor_observations: bool = True,
        spec: str | MjSpec = None,
        observation_spec: list[ObservationType] = None,
        actuation_spec: list[str] = None,
        mjx_backend: str = "jax",
        perturbation_params: dict | None = None,
        reflex_params: dict | None = None,
        **kwargs,
    ) -> None:
        """
        Constructor for MyoFullBody environment.

        Args:
            timestep (float): Simulation timestep in seconds. Default 0.002 (500Hz physics).
            n_substeps (int): Number of physics substeps per control step. Default 5 (100Hz control).
            disable_fingers (bool): If True (default), finger joints and muscles are disabled
            enable_joint_pos_observations (bool): If True (default), include joint positions in observations
            enable_joint_vel_observations (bool): If True (default), include joint velocities in observations
            enable_muscle_length_observations (bool): If True, include muscle length in observations
            enable_muscle_velocity_observations (bool): If True, include muscle velocity in observations
            enable_muscle_force_observations (bool): If True, include muscle force in observations
            enable_muscle_excitation_observations (bool): If True, include muscle excitation (neural drive from data.ctrl) in observations
            enable_muscle_activation_observations (bool): If True, include muscle activation (actual state from data.act) in observations
            spec (Union[str, MjSpec]): Path to XML file or MjSpec object. If None, uses default.
            observation_spec (List[ObservationType]): Custom observation specification.
            actuation_spec (List[str]): Custom action specification.
            **kwargs: Additional arguments passed to parent class.
        """

        self._disable_fingers = disable_fingers
        self._enable_joint_pos_observations = enable_joint_pos_observations
        self._enable_joint_vel_observations = enable_joint_vel_observations
        self._enable_muscle_length_observations = enable_muscle_length_observations
        self._enable_muscle_velocity_observations = enable_muscle_velocity_observations
        self._enable_muscle_force_observations = enable_muscle_force_observations
        self._enable_muscle_excitation_observations = enable_muscle_excitation_observations
        self._enable_muscle_activation_observations = enable_muscle_activation_observations
        self._enable_touch_sensor_observations = enable_touch_sensor_observations
        self._perturbation_config = PerturbationConfig(**(perturbation_params or {}))
        self._reflex_params = dict(reflex_params or {})
        self._reflex_enabled = bool(self._reflex_params.get("enabled", False))

        # Store mjx_backend for use in _modify_spec_for_mjx
        self.mjx_backend = mjx_backend

        # Store nconmax and num_envs from kwargs for automatic scaling
        self.nconmax = kwargs.get("nconmax", None)
        self.njmax = kwargs.get("njmax", None)  # Also handle njmax
        self.num_envs = kwargs.get("num_envs", 1)

        if spec is None:
            spec = self.get_default_xml_file_path()

        # Load the model specification
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        # Apply changes to the MjSpec
        spec = self._apply_spec_changes(spec)

        # Get observation and action specifications
        if observation_spec is None:
            observation_spec = self._get_observation_specification(spec)
        else:
            observation_spec = self.parse_observation_spec(observation_spec)

        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        # Modify spec for MJX if enabled
        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        super().__init__(
            timestep=timestep,
            n_substeps=n_substeps,
            spec=spec,
            actuation_spec=actuation_spec,
            observation_spec=observation_spec,
            mjx_backend=mjx_backend,
            **kwargs,
        )
        self._perturbation_dof_index = resolve_joint_dof_index(
            self._model, self._perturbation_config.joint, self._perturbation_config.side
        )
        if self._perturbation_config.enabled and self._perturbation_config.onset_mode == "stance_percentage":
            raise NotImplementedError(
                "Stance-percentage triggering requires a validated stance-duration estimator"
            )
        self._right_contact_sensor_addresses = tuple(
            int(self._model.sensor_adr[self._model.sensor(name).id]) for name in ("r_foot", "r_toes")
        )
        self._left_contact_sensor_addresses = tuple(
            int(self._model.sensor_adr[self._model.sensor(name).id]) for name in ("l_foot", "l_toes")
        )
        actuator_names = tuple(self._model.actuator(index).name for index in self._action_indices)
        self._reflex_groups = build_lower_body_groups(actuator_names)
        self._reflex_distribution = distribution_matrix(self._reflex_groups, len(actuator_names))
        joint_name_by_function = {
            "hip": "hip_flexion_{suffix}",
            "knee": "knee_angle_{suffix}",
            "ankle": "ankle_angle_{suffix}",
        }
        reflex_qpos_indices = []
        reflex_qvel_indices = []
        reflex_joint_scales = []
        for group in self._reflex_groups:
            joint_kind = group.function.split("_", 1)[0]
            suffix = "r" if group.side == "right" else "l"
            joint_id = self._model.joint(joint_name_by_function[joint_kind].format(suffix=suffix)).id
            reflex_qpos_indices.append(int(self._model.jnt_qposadr[joint_id]))
            reflex_qvel_indices.append(int(self._model.jnt_dofadr[joint_id]))
            joint_range = self._model.jnt_range[joint_id]
            reflex_joint_scales.append(max(float(joint_range[1] - joint_range[0]) * 0.5, 0.1))
        self._reflex_qpos_indices = np.asarray(reflex_qpos_indices, dtype=np.int32)
        self._reflex_qvel_indices = np.asarray(reflex_qvel_indices, dtype=np.int32)
        self._reflex_joint_scales = np.asarray(reflex_joint_scales, dtype=np.float32)
        self._reflex_right_group_mask = np.asarray(
            [group.side == "right" for group in self._reflex_groups], dtype=bool
        )
        self._reflex_pelvis_body_id = self._model.body("sacrum").id
        self._reflex_stance_gains = self._make_reflex_gains(self._reflex_params.get("stance_gains", {}))
        self._reflex_swing_gains = self._make_reflex_gains(self._reflex_params.get("swing_gains", {}))
        impulse_steps = int(np.ceil(self._perturbation_config.duration_s / self.dt))
        impulse_times = np.arange(impulse_steps, dtype=np.float64) * self.dt
        impulse_torque = torque_pulse(
            time_s=impulse_times,
            onset_s=np.asarray(0.0),
            duration_s=np.asarray(self._perturbation_config.duration_s),
            magnitude_nm=np.asarray(self._perturbation_config.magnitude_nm),
            direction=np.asarray(self._perturbation_config.direction),
            waveform=self._perturbation_config.waveform,
            enabled=self._perturbation_config.enabled,
            backend=np,
        )
        self._perturbation_impulse_nms = float(np.sum(impulse_torque) * self.dt)

    def _make_reflex_gains(self, values: dict) -> ReflexGains:
        """Expand scalar or per-group gain configuration outside compiled steps."""
        n_groups = len(self._reflex_groups)

        def expand(name):
            value = np.asarray(values.get(name, 0.0), dtype=np.float32)
            if value.ndim == 0:
                return np.full(n_groups, float(value), dtype=np.float32)
            if value.shape != (n_groups,):
                raise ValueError(f"reflex {name} gains must be scalar or have {n_groups} entries")
            return value

        return ReflexGains(*(expand(name) for name in ReflexGains._fields))

    def _init_additional_carry(self, key, model, data, backend):
        """Initialize fixed-shape reflex state for stable CPU and MJX pytrees."""
        carry = super()._init_additional_carry(key, model, data, backend)
        return carry.replace(
            previous_reflex_residual=backend.zeros(self.info.action_space.shape, dtype=backend.float32),
            reflex_group_action=backend.zeros(len(self._reflex_groups), dtype=backend.float32),
            reflex_saturation_fraction=backend.asarray(0.0, dtype=backend.float32),
        )

    def _apply_reflex_residual(self, action, data, carry, backend):
        """Add the grouped reflex in normalized policy-action space before control scaling."""
        if not self._reflex_enabled:
            return action, carry
        indices_q = backend.asarray(self._reflex_qpos_indices)
        indices_v = backend.asarray(self._reflex_qvel_indices)
        scales = backend.asarray(self._reflex_joint_scales)
        if self.th is not None:
            reference = self.th.get_current_traj_data(carry, backend)
            joint_error = (reference.qpos[indices_q] - data.qpos[indices_q]) / scales
            joint_velocity = (reference.qvel[indices_v] - data.qvel[indices_v]) / 10.0
        else:
            joint_error = -data.qpos[indices_q] / scales
            joint_velocity = -data.qvel[indices_v] / 10.0
        right_contact = backend.any(data.sensordata[backend.asarray(self._right_contact_sensor_addresses)] > 0.0)
        left_contact = backend.any(data.sensordata[backend.asarray(self._left_contact_sensor_addresses)] > 0.0)
        right_mask = backend.asarray(self._reflex_right_group_mask)
        stance_weight = backend.where(right_mask, right_contact, left_contact).astype(joint_error.dtype)
        limb_load = stance_weight
        rotation = data.xmat[self._reflex_pelvis_body_id].reshape(3, 3)
        pelvis_roll = backend.arctan2(rotation[2, 1], rotation[2, 2])
        pelvis_pitch = backend.arctan2(-rotation[2, 0], backend.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2))
        pelvis_angle = backend.where(right_mask, pelvis_roll, pelvis_pitch)
        pelvis_angular_velocity = backend.full_like(joint_error, data.cvel[self._reflex_pelvis_body_id, 1])
        stance_gains = ReflexGains(*(backend.asarray(value) for value in self._reflex_stance_gains))
        swing_gains = ReflexGains(*(backend.asarray(value) for value in self._reflex_swing_gains))
        group_action = compute_reflex_action(
            joint_error,
            joint_velocity,
            limb_load,
            pelvis_angle,
            pelvis_angular_velocity,
            stance_weight,
            stance_gains,
            swing_gains,
            float(self._reflex_params.get("group_limit", 0.25)),
            enabled=True,
        )
        residual = distribute_group_action(group_action, backend.asarray(self._reflex_distribution))
        muscle_limit = float(self._reflex_params.get("muscle_limit", 0.25))
        residual = backend.clip(residual, -muscle_limit, muscle_limit)
        previous = carry.previous_reflex_residual
        max_delta = float(self._reflex_params.get("rate_limit_per_s", 5.0)) * self.dt
        residual = previous + backend.clip(residual - previous, -max_delta, max_delta)
        scale = float(self._reflex_params.get("scale", 1.0))
        unclipped = action + scale * residual
        composed = backend.clip(unclipped, -1.0, 1.0)
        saturation = backend.mean((unclipped < -1.0) | (unclipped > 1.0))
        return composed, carry.replace(
            previous_reflex_residual=residual,
            reflex_group_action=group_action,
            reflex_saturation_fraction=saturation,
        )

    def _preprocess_action(self, action, model, data, carry):
        action, carry = super()._preprocess_action(action, model, data, carry)
        return self._apply_reflex_residual(action, data, carry, np)

    def _mjx_preprocess_action(self, action, model, data, carry):
        action, carry = super()._mjx_preprocess_action(action, model, data, carry)
        return self._apply_reflex_residual(action, data, carry, jnp)

    def _update_perturbation(self, data, carry, backend):
        """Update one-shot state and compute external torque for this step."""
        config = self._perturbation_config
        step = backend.asarray(carry.cur_step_in_episode - 1, dtype=backend.int32)
        phase_fraction = None
        if config.onset_mode == "phase":
            if self.th is None:
                raise ValueError("Reference-phase triggering requires a loaded trajectory")
            trajectory_length = self.th.len_trajectory(carry.traj_state.traj_no)
            phase_fraction = carry.traj_state.subtraj_step_no / backend.maximum(trajectory_length - 1, 1)
        right_contact = backend.any(data.sensordata[backend.asarray(self._right_contact_sensor_addresses)] > 0.0)
        left_contact = backend.any(data.sensordata[backend.asarray(self._left_contact_sensor_addresses)] > 0.0)
        selected_contact = right_contact if config.side == "right" else left_contact
        previous_contact = (
            carry.previous_right_foot_contact if config.side == "right" else carry.previous_left_foot_contact
        )
        heel_strike_event = selected_contact & ~previous_contact
        state = PerturbationState(carry.perturbation_triggered, carry.perturbation_onset_step)
        state, _ = update_perturbation_trigger(
            state,
            step,
            self.dt,
            config.onset_mode,
            onset_time_s=config.onset_time_s,
            phase_fraction=phase_fraction,
            onset_phase=config.onset_phase,
            phase_tolerance=config.phase_tolerance,
            heel_strike_event=heel_strike_event,
        )
        state = PerturbationState(
            triggered=backend.where(config.enabled, state.triggered, carry.perturbation_triggered),
            onset_step=backend.where(config.enabled, state.onset_step, carry.perturbation_onset_step),
        )
        torque_nm = triggered_torque(
            state, step, self.dt, config.duration_s, config.magnitude_nm, config.direction, config.waveform
        )
        torque_nm = backend.where(config.enabled, torque_nm, backend.asarray(0.0))
        carry = carry.replace(
            perturbation_triggered=state.triggered,
            perturbation_onset_step=state.onset_step,
            previous_right_foot_contact=right_contact,
            previous_left_foot_contact=left_contact,
        )
        return carry, torque_nm

    def _simulation_pre_step(self, model, data, carry):
        """Apply CPU MuJoCo exoskeleton torque separately from muscle controls."""
        model, data, carry = super()._simulation_pre_step(model, data, carry)
        carry, torque_nm = self._update_perturbation(data, carry, np)
        data.qfrc_applied[self._perturbation_dof_index] = float(torque_nm)
        return model, data, carry

    def _mjx_simulation_pre_step(self, model, data, carry):
        """Apply JAX-vectorizable exoskeleton torque separately from `data.ctrl`."""
        model, data, carry = super()._mjx_simulation_pre_step(model, data, carry)
        carry, torque_nm = self._update_perturbation(data, carry, jnp)
        qfrc_applied = data.qfrc_applied.at[self._perturbation_dof_index].set(torque_nm)
        return model, data.replace(qfrc_applied=qfrc_applied), carry

    def _update_info_dictionary(self, info, obs, data, carry):
        """Log CPU perturbation diagnostics in SI units."""
        info = super()._update_info_dictionary(info, obs, data, carry)
        info["perturbation_torque_nm"] = float(data.qfrc_applied[self._perturbation_dof_index])
        info["perturbation_requested_torque_nm"] = (
            self._perturbation_config.direction * self._perturbation_config.magnitude_nm
        )
        info["perturbation_dof_index"] = self._perturbation_dof_index
        onset_s = float(carry.perturbation_onset_step) * self.dt
        info["perturbation_onset_s"] = onset_s if carry.perturbation_onset_step >= 0 else -1.0
        info["perturbation_stop_s"] = (
            onset_s + self._perturbation_config.duration_s if carry.perturbation_onset_step >= 0 else -1.0
        )
        info["perturbation_impulse_nms"] = self._perturbation_impulse_nms
        info["reflex_enabled"] = self._reflex_enabled
        info["reflex_saturation_fraction"] = float(carry.reflex_saturation_fraction)
        info["reflex_group_action"] = (
            np.zeros(len(self._reflex_groups), dtype=np.float32)
            if carry.reflex_group_action is None
            else np.asarray(carry.reflex_group_action)
        )
        return info

    def _mjx_update_info_dictionary(self, info, obs, data, carry):
        """Log MJX perturbation diagnostics without leaving traced array code."""
        info = super()._mjx_update_info_dictionary(info, obs, data, carry)
        info["perturbation_torque_nm"] = data.qfrc_applied[self._perturbation_dof_index]
        info["perturbation_requested_torque_nm"] = jnp.asarray(
            self._perturbation_config.direction * self._perturbation_config.magnitude_nm
        )
        info["perturbation_dof_index"] = jnp.asarray(self._perturbation_dof_index, dtype=jnp.int32)
        onset_s = carry.perturbation_onset_step * self.dt
        info["perturbation_onset_s"] = jnp.where(carry.perturbation_onset_step >= 0, onset_s, -1.0)
        info["perturbation_stop_s"] = jnp.where(
            carry.perturbation_onset_step >= 0,
            onset_s + self._perturbation_config.duration_s,
            -1.0,
        )
        info["reflex_enabled"] = jnp.asarray(self._reflex_enabled)
        info["reflex_saturation_fraction"] = carry.reflex_saturation_fraction
        info["reflex_group_action"] = (
            jnp.zeros(len(self._reflex_groups), dtype=jnp.float32)
            if carry.reflex_group_action is None
            else carry.reflex_group_action
        )
        info["perturbation_impulse_nms"] = jnp.asarray(self._perturbation_impulse_nms)
        return info

    def _apply_spec_changes(self, spec: MjSpec) -> MjSpec:
        """
        Apply changes to the MjSpec including:
        1. Disabling fingers if requested (same as MyoBimanualArm)
        2. Adding mimic sites for trajectory tracking
        3. Modifying muscle control ranges from [0, 1] to [-1, 1] to match MyoBimanualArm convention

        Args:
            spec (MjSpec): The MuJoCo model specification

        Returns:
            MjSpec: Modified specification
        """

        # Handle finger disabling if requested (same logic as MyoBimanualArm)
        if self._disable_fingers:
            # Define specific finger joint names to avoid matching hip joints
            # Use the exact finger joint names from MyoBimanualArm
            finger_joints = [
                # Right hand finger joints (from myoarm_body.xml)
                "cmc_flexion_r",
                "cmc_abduction_r",
                "mp_flexion_r",
                "ip_flexion_r",
                "mcp2_flexion_r",
                "mcp2_abduction_r",
                "mcp3_flexion_r",
                "mcp3_abduction_r",
                "mcp4_flexion_r",
                "mcp4_abduction_r",
                "mcp5_flexion_r",
                "mcp5_abduction_r",
                "md2_flexion_r",
                "md3_flexion_r",
                "md4_flexion_r",
                "md5_flexion_r",
                "pm2_flexion_r",
                "pm3_flexion_r",
                "pm4_flexion_r",
                "pm5_flexion_r",
                # Left hand finger joints (from myoarm_left_body.xml - uses "L" suffix)
                "cmc_flexion_l",
                "cmc_abduction_l",
                "mp_flexion_l",
                "ip_flexion_l",
                "mcp2_flexion_l",
                "mcp2_abduction_l",
                "mcp3_flexion_l",
                "mcp3_abduction_l",
                "mcp4_flexion_l",
                "mcp4_abduction_l",
                "mcp5_flexion_l",
                "mcp5_abduction_l",
                "md2_flexion_l",
                "md3_flexion_l",
                "md4_flexion_l",
                "md5_flexion_l",
                "pm2_flexion_l",
                "pm3_flexion_l",
                "pm4_flexion_l",
                "pm5_flexion_l",
            ]

            finger_muscles = [
                # Right hand muscles
                "FDS2",
                "FDS3",
                "FDS4",
                "FDS5",  # Finger flexors (superficial)
                "FDP2",
                "FDP3",
                "FDP4",
                "FDP5",  # Finger flexors (deep)
                "EDC2",
                "EDC3",
                "EDC4",
                "EDC5",  # Finger extensors
                "EDM",
                "EIP",  # Finger extensors (specific)
                "EPL",
                "EPB",
                "FPL",
                "APL",  # Thumb muscles
                "OP",  # Opponens pollicis
                "RI2",
                "RI3",
                "RI4",
                "RI5",  # Radial interossei
                "LU_RB2",
                "LU_RB3",
                "LU_RB4",
                "LU_RB5",  # Lumbricals
                "UI_UB2",
                "UI_UB3",
                "UI_UB4",
                "UI_UB5",  # Ulnar interossei
                # Left hand muscles (with L suffix)
                "FDS2_left",
                "FDS3_left",
                "FDS4_left",
                "FDS5_left",  # Left finger flexors (superficial)
                "FDP2_left",
                "FDP3_left",
                "FDP4_left",
                "FDP5_left",  # Left finger flexors (deep)
                "EDC2_left",
                "EDC3_left",
                "EDC4_left",
                "EDC5_left",  # Left finger extensors
                "EDM_left",
                "EIP_left",  # Left finger extensors (specific)
                "EPL_left",
                "EPB_left",
                "FPL_left",
                "APL_left",  # Left thumb muscles
                "OP_left",  # Left opponens pollicis
                "RI2_left",
                "RI3_left",
                "RI4_left",
                "RI5_left",  # Left radial interossei
                "LU_RB2_left",
                "LU_RB3_left",
                "LU_RB4_left",
                "LU_RB5_left",  # Left lumbricals
                "UI_UB2_left",
                "UI_UB3_left",
                "UI_UB4_left",
                "UI_UB5_left",  # Left ulnar interossei
            ]

            # Remove finger joints (use exact match to avoid matching hip joints)
            joints_to_remove = []
            for joint in spec.joints:
                if joint.name in finger_joints:
                    joints_to_remove.append(joint)

            for joint in joints_to_remove:
                spec.delete(joint)
            # print(f"[MyoFullBody] Removed {len(joints_to_remove)} finger joints: {[j.name for j in joints_to_remove]}")

            # Remove finger muscles and their tendons (use exact match to avoid matching arm muscles)
            actuators_to_remove = []
            for actuator in spec.actuators:
                if actuator.name in finger_muscles:
                    actuators_to_remove.append(actuator)

            # print(f"[MyoFullBody] Removing {len(actuators_to_remove)} finger muscles: {[a.name for a in actuators_to_remove[:10]]}")
            for actuator in actuators_to_remove:
                spec.delete(actuator)

            # Remove associated tendons (use clean substring matching like MyoBimanualArm)
            tendons_to_remove = []
            for tendon in spec.tendons:
                if any(finger_muscle in tendon.name for finger_muscle in finger_muscles):
                    tendons_to_remove.append(tendon)

            # print(f"[MyoFullBody] Removing {len(tendons_to_remove)} finger tendons: {[t.name for t in tendons_to_remove[:10]]}")
            for tendon in tendons_to_remove:
                spec.delete(tendon)

        # Add mimic sites for trajectory tracking
        for body_name, site_name in self.body2sites_for_mimic.items():
            b = spec.body(body_name)
            pos = [0.0, 0.0, 0.0]
            b.add_site(
                name=site_name,
                group=4,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.075, 0.05, 0.025],
                rgba=[1.0, 0.0, 0.0, 0.5],
                pos=pos,
            )

        
        # Modify control range to -1 to 1
        for actuator in spec.actuators:
            # Only modify muscle actuators (type 4 in MuJoCo), not motor actuators
            if actuator.dyntype == mujoco.mjtDyn.mjDYN_MUSCLE:
                actuator.ctrlrange = [-1.0, 1.0]
                actuator.ctrllimited = True
        return spec

    def _get_observation_specification(self, spec: MjSpec) -> list[ObservationType]:
        """
        Get observation specification including joint positions/velocities
        and optional muscle observations.

        Args:
            spec (MjSpec): The MuJoCo model specification

        Returns:
            List[ObservationType]: List of observations
        """
        obs_spec = []

        # Get all joint names except the root
        j_names = [j.name for j in spec.joints if j.name != self.root_free_joint_xml_name]

        # Add joint position observations if enabled
        if self._enable_joint_pos_observations:
            # Add free joint observation (position without x,y)
            obs_spec.append(ObservationType.FreeJointPosNoXY("q_free_joint", self.root_free_joint_xml_name))
            # Add all joint positions
            obs_spec.append(ObservationType.JointPosArray("q_all_pos", j_names))

        # Add joint velocity observations if enabled
        if self._enable_joint_vel_observations:
            # Add free joint velocities
            obs_spec.append(ObservationType.FreeJointVel("dq_free_joint", self.root_free_joint_xml_name))
            # Add all joint velocities
            obs_spec.append(ObservationType.JointVelArray("dq_all_vel", j_names))

        # Add muscle observations if enabled
        for actuator in spec.actuators:
            actuator_name = actuator.name

            # Add muscle length observations
            if self._enable_muscle_length_observations:
                obs_name = f"muscle_length_{actuator_name.lower()}"
                obs_spec.append(ObservationType.ActuatorLength(obs_name, xml_name=actuator_name))

            # Add muscle velocity observations
            if self._enable_muscle_velocity_observations:
                obs_name = f"muscle_velocity_{actuator_name.lower()}"
                obs_spec.append(ObservationType.ActuatorVelocity(obs_name, xml_name=actuator_name))

            # Add muscle force observations
            if self._enable_muscle_force_observations:
                obs_name = f"muscle_force_{actuator_name.lower()}"
                obs_spec.append(ObservationType.ActuatorForce(obs_name, xml_name=actuator_name))

            # Add muscle excitation observations (neural drive from data.ctrl)
            if self._enable_muscle_excitation_observations:
                obs_name = f"muscle_excitation_{actuator_name.lower()}"
                obs_spec.append(ObservationType.ActuatorExcitation(obs_name, xml_name=actuator_name))

            # Add muscle activation observations (actual state from data.act)
            if self._enable_muscle_activation_observations:
                obs_name = f"muscle_activation_{actuator_name.lower()}"
                obs_spec.append(ObservationType.ActuatorActivation(obs_name, xml_name=actuator_name))

        # Add touch sensor observations for foot contact feedback if enabled
        # These sensors are crucial for locomotion balance and gait control
        if self._enable_touch_sensor_observations:
            touch_sensors = ["r_foot", "r_toes", "l_foot", "l_toes"]
            for sensor_name in touch_sensors:
                obs_name = f"touch_{sensor_name}"
                obs_spec.append(ObservationType.TouchSensor(obs_name, xml_name=sensor_name))

        return obs_spec

    def _get_action_specification(self, spec: MjSpec) -> list[str]:
        """
        Get action specification - returns all actuator names.

        Args:
            spec (MjSpec): The MuJoCo model specification

        Returns:
            List[str]: List of actuator names
        """
        action_spec = []
        for actuator in spec.actuators:
            action_spec.append(actuator.name)
        return action_spec

    def _modify_spec_for_mjx(self, spec: MjSpec) -> MjSpec:
        """
        Modify the model specification for MJX backend compatibility.
        Uses the same contact-simplification idea as the older full-body setup,
        adapted for the maintained MyoFullBody asset.
        """
        if hasattr(self, "mjx_backend") and self.mjx_backend == "warp":
            # MuJoCo 3.3.7+: nconmax is per-env, naconmax is total across all envs
            per_env_contacts = 96  # Contacts per environment
            num_envs = getattr(self, "num_envs", 1)

            if not hasattr(self, "nconmax") or self.nconmax is None:
                self.nconmax = per_env_contacts  # Per-env contacts
            if not hasattr(self, "naconmax") or getattr(self, "naconmax", None) is None:
                self.naconmax = self.nconmax * num_envs  # Total across all envs
            if not hasattr(self, "njmax") or self.njmax is None:
                self.njmax = 768  # Per-world constraints

            # Apply the limits to the spec (spec uses per-env values)
            spec.nconmax = self.nconmax
            spec.njmax = self.njmax

            logger.info(
                "nconmax=%s (per-env), naconmax=%s (total), njmax=%s",
                self.nconmax,
                self.naconmax,
                self.njmax,
            )
            logger.info("Keeping all contacts enabled for Warp backend")
        else:
            for g in spec.geoms:
                # Keep essential ground contacts but disable others
                if g.name and ("floor" in g.name.lower() or "ground" in g.name.lower()):
                    continue  # Keep floor/ground contacts
                g.contype = 0
                g.conaffinity = 0

        return spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default path to the xml file of the environment.
        """
        return musclemimic_models.get_xml_path("myofullbody").as_posix()

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the root free joint in the Mujoco xml file.
        """
        return "root"

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body in the Mujoco xml file.
        """
        return "pelvis"

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body in the Mujoco xml file.
        """
        return "torso"

    @info_property
    def root_height_healthy_range(self) -> tuple[float, float]:
        """
        Returns the healthy range of the root height.
        """
        return (0.6, 1.5)

    @info_property
    def body2sites_for_mimic(self) -> dict[str, str]:
        """
        Returns a dictionary mapping body names to their corresponding mimic site names.
        Tailored for the maintained full-body muscle model.

        Focus on essential tracking points: ankle and toe for proper foot orientation.
        Let forward kinematics handle heel (calcn) positioning.
        """
        body2sitemimic = {
            "pelvis": "pelvis_mimic",
            "lumbar1": "upper_body_mimic",
            "head": "head_mimic",
            # Left arm (note: left uses capital L suffix)
            "humerus_l": "left_shoulder_mimic",
            "ulna_l": "left_elbow_mimic",
            "lunate_l": "left_hand_mimic",
            # Right arm (note: right has no suffix)
            "humerus_r": "right_shoulder_mimic",
            "ulna_r": "right_elbow_mimic",
            "lunate_r": "right_hand_mimic",
            "femur_l": "left_hip_mimic",
            "tibia_l": "left_knee_mimic",
            "talus_l": "left_ankle_mimic",
            "toes_l": "left_toes_mimic",
            "femur_r": "right_hip_mimic",
            "tibia_r": "right_knee_mimic",
            "talus_r": "right_ankle_mimic",
            "toes_r": "right_toes_mimic",
        }
        return body2sitemimic

    @info_property
    def sites_for_mimic(self) -> list[str]:
        """
        Returns a list of all mimic sites.
        """
        return list(self.body2sites_for_mimic.values())

    @info_property
    def goal_visualization_arrow_offset(self) -> list[float]:
        """
        Returns the offset for the goal visualization arrow.
        """
        return [0, 0, 0.4]


class MjxMyoFullBody(MyoFullBody):
    """
    MJX version of MyoFullBody with support for JAX and warp backends.
    """

    mjx_enabled = True

    def __init__(self, timestep: float = 0.002, n_substeps: int = 5, mjx_backend: str = "jax", **kwargs):
        """
        Constructor for MJX version of MyoFullBody.

        Args:
            timestep (float): Timestep of the simulation.
            n_substeps (int): Number of substeps.
            mjx_backend (str): MJX backend to use ('jax' or 'warp'). Default: 'jax'.
            **kwargs: Additional arguments.
        """
        # Extract goal-related parameters to prevent them from being passed to viewer
        goal_related_params = [
            "visualize_goal",
            "enable_enhanced_visualization",
            "target_geom_rgba",
            "n_step_lookahead",
            "goal_type",
            "goal_params",
        ]

        extracted_goal_params = {}
        for param in goal_related_params:
            if param in kwargs:
                extracted_goal_params[param] = kwargs.pop(param)

        if "model_option_conf" not in kwargs.keys():
            model_option_conf = dict(iterations=4, ls_iterations=8, disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        else:
            model_option_conf = kwargs["model_option_conf"]
            del kwargs["model_option_conf"]

        # Pass goal-related parameters back through kwargs
        kwargs.update(extracted_goal_params)

        # Store mjx_backend for use in parent class
        self.mjx_backend = mjx_backend

        super().__init__(
            timestep=timestep,
            n_substeps=n_substeps,
            model_option_conf=model_option_conf,
            mjx_backend=mjx_backend,
            **kwargs,
        )
