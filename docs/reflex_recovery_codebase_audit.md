# Reflex Recovery Codebase Audit

Date: 2026-07-17

Branch: `research/reflex-recovery`

Audited commit: `8a2e1a7ff1c4cdcea64f9e25526b805c52696df2`

Working tree note: the checkout already had local changes before this audit:

- `musclemimic/runner/engine.py` modified
- `fullbody/conf_lowerbody_walking_disturbance.yaml` untracked

This audit is intentionally conservative. It records verified extension points and avoids claiming that the baseline checkpoint, model toe structure, or MuJoCo Warp execution has been reproduced until those checks are run on the target machine.

## Baseline Reproduction Record

The initial deterministic baseline was reproduced on 2026-07-17 on an NVIDIA H200 using checkpoint `amathislab/mm-10m-2` and official retargeted motion `KIT/314/walking_medium09_poses`.

- evaluation seed: `0`
- action mode: deterministic distribution mean
- physics timestep: `0.002 s`
- policy timestep: `0.01 s` (five physics substeps)
- reference frequency: `100 Hz`
- evaluated steps: `200`
- total upstream reward: `252.847`
- policy-network output range: `[-7.940, 5.730]`
- applied muscle-command range after `DefaultControl`: `[-1.000, 1.000]`
- actuator dimension: `354`, matching the compiled model
- exported rollout: `outputs/reflex_recovery/baseline/myofullbody_episodes_mujoco_20260717_152806.npz`

An identical second run was saved under `outputs/reflex_recovery/baseline_repro`. Joint positions, velocities, accelerations, touch observations, raw policy actions, applied muscle commands, muscle activations, rewards, and simulation timestamps were bitwise identical. Wall-clock duration and FPS were intentionally excluded from reproducibility comparison.

The checkpoint policy distribution itself is unbounded. Therefore the phrase “normalized policy output in `[-1, 1]`” below must be interpreted as the controller input convention, not a guaranteed bound on raw neural-network samples. `DefaultControl.generate_action` clips/maps the raw policy action before the command reaches MuJoCo. Residual injection must explicitly enforce normalized limits before calling that control function.

## Repository Entry Points

The full-body training entrypoint is `fullbody/experiment.py`. It is a Hydra application that calls `musclemimic.runner.engine.run_experiment(config, hooks=UnifiedHooks())`.

The full-body evaluation entrypoint is `fullbody/eval.py`. It restores a checkpoint with `musclemimic.runner.eval_utils.load_checkpoint`, reconstructs the environment through `loco_mujoco.task_factories.TaskFactory`, creates a PPO agent configuration with `musclemimic.algorithms.PPOJax.init_agent_conf`, and then dispatches to MuJoCo/MJX playback, rollout export, viewer playback, or validation metrics helpers.

Documented released-checkpoint evaluation currently uses:

```bash
uv run mjpython fullbody/eval.py \
  --path hf://amathislab/mm-10m-2 \
  --motion_path KIT/314/walking_medium09_poses \
  --use_mujoco \
  --stochastic \
  --eval_seed 0 \
  --n_steps 1000 \
  --mujoco_viewer
```

The README also lists the MyoFullBody base checkpoint resource as `hf://amathislab/mm-fullbody-base`. The study configuration should keep checkpoint and motion paths configurable because both names appear in upstream documentation.

## Environment Class

MyoFullBody is implemented in `musclemimic/environments/humanoids/myofullbody.py` as `MyoFullBody(LocoEnv)`.

The class is registered through the environment registration imported by `musclemimic.environments`. Environment construction is indirect:

- `musclemimic.runner.engine.instantiate_env` gets the configured factory class from `loco_mujoco.task_factories.TaskFactory`.
- `loco_mujoco/task_factories/imitation_factory.py::ImitationFactory.make` validates `env_name`, constructs `LocoEnv.registered_envs[env_name]`, and loads trajectory data.
- `loco_mujoco/task_factories/rl_factory.py::RLFactory.make` constructs registered environments for non-imitation RL tasks.

Full-body imitation configs use `fullbody/conf_fullbody.yaml` plus GMR overrides in `fullbody/conf_fullbody_gmr.yaml`, `fullbody/conf_fullbody_demo.yaml`, and `fullbody/conf_fullbody_gmr_resnet.yaml`.

## Reset and Step Functions

CPU MuJoCo reset and step are in `loco_mujoco/core/mujoco_base.py`:

- `Mujoco.reset`
- `Mujoco.step`

MJX reset and step are in `musclemimic/core/mujoco_mjx.py`:

- `Mjx.mjx_reset`
- `Mjx.mjx_step`

`LocoEnv` in `musclemimic/environments/base.py` extends these paths with trajectory state updates:

- `LocoEnv._reset_carry`
- `LocoEnv._mjx_reset_carry`
- `LocoEnv._simulation_post_step`
- `LocoEnv._mjx_simulation_post_step`
- `LocoEnv._is_done`
- `LocoEnv._mjx_is_done`

The MJX step path is scan-based and JIT-compatible:

1. Store `last_action` in carry.
2. Call `_mjx_preprocess_action`.
3. Call `_mjx_simulation_pre_step`.
4. Inside `jax.lax.scan`, call `_mjx_compute_action`.
5. Write the computed control into `data.ctrl` at `self._action_indices`.
6. Call `mjx.step` for `self._n_substeps`.
7. Call `_mjx_simulation_post_step`.
8. Build observation, reward, absorbing, done, and info.

## Observation Construction

The observation specification for MyoFullBody is built by `MyoFullBody._get_observation_specification` in `musclemimic/environments/humanoids/myofullbody.py`.

Default observations include:

- root free-joint position without x/y via `ObservationType.FreeJointPosNoXY`
- all non-root joint positions via `ObservationType.JointPosArray`
- root free-joint velocity via `ObservationType.FreeJointVel`
- all non-root joint velocities via `ObservationType.JointVelArray`
- foot touch sensors `r_foot`, `r_toes`, `l_foot`, `l_toes` when touch observations are enabled

Optional muscle observations are exposed through flags:

- `enable_muscle_length_observations`
- `enable_muscle_velocity_observations`
- `enable_muscle_force_observations`
- `enable_muscle_excitation_observations`
- `enable_muscle_activation_observations`

Muscle sensory feedback must therefore be treated as optional and discovered from the active configuration, not assumed to exist in every observation.

## Action Transformation and Actuator Command Path

The action specification for MyoFullBody is built by `MyoFullBody._get_action_specification`, which returns all actuator names in the compiled spec.

`MyoFullBody._apply_spec_changes` sets every MuJoCo muscle actuator control range to `[-1.0, 1.0]` and enables `ctrllimited`.

Action indices are resolved in `loco_mujoco/core/mujoco_base.py::Mujoco.get_action_indices`. If `actuation_spec` is empty, all actuators are controlled; otherwise named actuators are mapped with `model.actuator(name).id`.

The default control path is `loco_mujoco/core/control_functions/default.py::DefaultControl`.

Confirmed action semantics:

- The PPO policy output is in normalized action space with limits `[-1, 1]`.
- `DefaultControl.generate_action` maps normalized action to actuator control range with `ctrl = action * norm_act_delta + norm_act_mean`.
- In `direct` mode, the mapped control is clipped to actuator limits.
- In `incremental` mode, clipped normalized actions are integrated against the previous actuator-space control and clipped to actuator limits.
- The final actuator command reaches MuJoCo through `data.ctrl[self._action_indices] = ctrl_action`.

Safest initial residual-injection point:

- Compose `u_base + alpha * u_reflex + beta * u_recovery` in normalized policy action space before `DefaultControl.generate_action`.
- Clip the composed normalized action to the policy action limits `[-1, 1]`.
- Let `DefaultControl` perform the existing actuator-space scaling and actuator `ctrlrange` clipping.

This avoids bypassing upstream actuator limit handling and keeps residual diagnostics independent of the MuJoCo control function.

## Reward Calculation

The MJX reward call is `musclemimic/core/mujoco_mjx.py::Mjx._mjx_reward`, which delegates to `self._reward_function`.

The CPU reward call is `loco_mujoco/core/mujoco_base.py::Mujoco._reward`, which delegates to the same reward-function abstraction.

Full-body imitation uses trajectory rewards from `musclemimic/core/reward/trajectory_based.py`.

Important classes:

- `TrajectoryBasedReward`
- `TargetVelocityTrajReward`
- `MimicReward`
- `MimicRewardState`

`MimicReward` computes DeepMimic-style tracking terms using current simulation state, current trajectory state, relative site quantities, root velocity, action costs, action-rate costs, and optional dynamics penalties. It also returns reward component info for logging.

Recovery-priority reward should wrap or subclass the existing reward function and preserve its component logging. It should not replace `MimicReward` wholesale.

## Termination Calculation

Generic MJX done logic is in `musclemimic/core/mujoco_mjx.py::Mjx._mjx_is_done`, which terminates on horizon or absorbing state. `LocoEnv._mjx_is_done` adds trajectory-end termination and goal-driven termination.

Full-body enhanced termination is implemented in `musclemimic/core/terminal_state_handler/enhanced_fullbody.py`.

Important classes:

- `EnhancedFullBodyTerminalStateHandler`
- `MeanSiteDeviationTerminalStateHandler`
- `MeanRelativeSiteDeviationWithRootTerminalStateHandler`

`EnhancedFullBodyTerminalStateHandler` combines height-based termination with optional ankle/root site deviation checks against reference trajectory data.

Recovery-aware termination should retain true fall/invalid-state checks but relax reference-deviation termination as a function of the recovery gate. The narrow extension point is a new terminal handler class rather than modifying existing defaults.

## Reference-Motion Phase Representation

Trajectory state is stored in `LocoCarry.traj_state` in `musclemimic/environments/base.py`.

`LocoEnv` updates trajectory state through `TrajectoryHandler.update_state` during post-step. State fields used elsewhere include:

- `traj_no`
- `subtraj_step_no`

Trajectory loading and current-frame access happen through `loco_mujoco/trajectory/handler.py::TrajectoryHandler` and calls such as:

- `env.th.get_current_traj_data(carry, backend)`
- `env.th.get_init_traj_data(carry, backend)`
- `env.th.len_trajectory(traj_no)`
- `env.th.reached_trajectory_end(traj_state, backend)`

Normalized phase should be derived from `subtraj_step_no / len_trajectory(traj_no)` unless a more explicit phase signal is found in the loaded trajectory metadata.

## Checkpoint Restore Mechanism

Evaluation checkpoint restore starts in `musclemimic/runner/eval_utils.py::load_checkpoint`.

That function:

1. canonicalizes local or Hugging Face paths with `musclemimic.runner.checkpointing._canonicalize_resume_path`;
2. constructs `musclemimic.algorithms.common.checkpoint_manager.UnifiedCheckpointManager`;
3. loads the checkpoint;
4. recreates config with `OmegaConf.create`;
5. converts restored state with `musclemimic.algorithms.ppo.checkpoint.create_agent_state_from_orbax`.

Training resume is handled in `musclemimic/runner/engine.py::run_experiment` via:

- `find_latest_checkpoint`
- `resolve_checkpoint_dir`
- `resume_or_fresh`
- `validate_checkpoint_compatibility`
- `write_manifest`

Checkpoint saving uses `musclemimic.algorithms.common.checkpoint_hooks.create_jax_checkpoint_host_callback` and Orbax managers in `musclemimic/algorithms/common/checkpoint_manager.py`.

## Policy Architecture and Parameter Tree

PPO creates the policy/value network in `musclemimic/algorithms/ppo/ppo.py::PPOJax._create_network`.

The default network is `musclemimic.algorithms.common.networks.ActorCritic`.

Important architecture fields:

- `actor_hidden_layers`
- `critic_hidden_layers`
- `activation`
- `init_std`
- `learnable_std`
- `actor_obs_group`
- `critic_obs_group`
- `use_residual`
- `residual_type`
- `residual_gate_init`

`ActorCritic.__call__` returns a `distrax.MultivariateNormalDiag` policy distribution and a scalar critic value. Actions are sampled from or taken as the mean of that distribution in PPO inference/training helpers.

For the recovery study, the frozen base policy should be restored as a separate train state or parameter subtree and called under `jax.lax.stop_gradient`. A bitwise-unchanged test should compare the base parameter tree before and after optimizer updates.

## PPO Update Loop

PPO training is implemented in `musclemimic/algorithms/ppo/runner.py::train`.

Important internal steps:

- wrap environment with `wrap_env`
- initialize train state with `_init_train_state`
- reset vectorized environments
- collect trajectories through `_collect_trajectories`
- compute GAE through `musclemimic.rl_core.compute_gae`
- create minibatches through `musclemimic.rl_core.create_minibatches`
- update parameters with PPO losses from `musclemimic/algorithms/ppo/loss.py`
- optionally update adaptive termination, reward curriculum, validation, metrics, and checkpoints

Residual PPO should reuse this runner only after the residual environment/policy interface is clear. The initial minimal path should first provide pure composition, reflex, perturbation, gate, and smoke-rollout utilities.

## Metric Logging and Video Evaluation

Training logging is assembled in `musclemimic/runner/engine.py::build_logging_callback`.

Validation metrics use:

- `musclemimic.utils.metrics.MetricsHandler`
- `musclemimic.utils.metrics.ValidationSummary`
- `musclemimic.runner.eval_utils.run_validation_metrics`
- `musclemimic.runner.eval_utils.run_validation_metrics_mjx_all`
- `musclemimic.runner.eval_utils.run_validation_metrics_mujoco`

Validation video recording is triggered from `build_logging_callback` using the hooks-provided recorder.

Evaluation video/viewer utilities live in `musclemimic/runner/eval_utils.py`, including:

- `run_with_mujoco_viewer`
- `run_with_trajectory_export`
- `PPOJax.play_policy`
- `PPOJax.play_policy_mujoco`

## External Force or Perturbation Utilities

No existing perturbation subsystem was identified in the audited core files. Domain randomization exists in `loco_mujoco/core/domain_randomizer`, but it is not an erroneous exoskeleton torque-impulse interface.

The disturbance should remain separate from human muscle actions. The likely MJX application point is before `mjx.step`, by adding to generalized forces (`qfrc_applied`) or a verified equivalent field. This must be confirmed against the active MuJoCo/MJX data fields before implementation.

## MuJoCo Warp Versus Standard MuJoCo State

CPU MuJoCo uses `mujoco.MjModel` and `mujoco.MjData`.

MJX uses `mujoco.mjx.Model` and `mujoco.mjx.Data`.

`musclemimic/core/mujoco_mjx.py::Mjx.__init__` builds `self.sys = mjx.put_model(self._model, impl=backend_impl)`.

For the `warp` backend, `mjx.make_data` is used because Warp cannot build `Data` from `MjData` with `mjx.put_data`. The code also tracks contact budgets:

- `nconmax`
- `njmax`
- `naconmax`

The standard JAX backend uses `mjx.put_data(self._model, self._data, impl=backend_impl)`.

Perturbation and residual logic must therefore be expressed as pure updates on `mjx.Data` fields and should not assume all CPU `MjData` mutation patterns are available under Warp.

## JIT-Compiled Components

`musclemimic/runner/engine.py::run_training` JIT-compiles the built train function. Multi-seed training uses `jax.vmap` plus `jax.jit`.

The MJX environment step uses `jax.lax.scan` in `Mjx.mjx_step`.

Vectorized environment wrappers in `musclemimic/core/wrappers/mjx.py` use vectorized reset/step paths and autoreset logic.

New reflex, gate, perturbation, reward, and action-composition code must avoid Python loops in traced step code and should expose JAX-compatible array APIs.

## Proposed Minimal Patch Plan

The first patch series should be additive and should not modify upstream defaults.

1. Add `musclemimic/research/reflex_recovery/` with pure, tested modules:
   - `config.py`
   - `delay_buffer.py`
   - `muscle_groups.py`
   - `reflex_controller.py`
   - `perturbations.py`
   - `stability.py`
   - `recovery_gate.py`
   - `action_composer.py`
   - `metrics.py`
2. Add fast unit tests under `tests/reflex_recovery/` for:
   - delay-buffer reset and read behavior
   - group-action distribution
   - zero reflex output when disabled or gains are zero
   - action clipping/rate diagnostics in normalized action space
   - perturbation waveform timing, sign, and impulse
   - recovery gate monotonicity and hysteresis/low-pass behavior
3. Add `scripts/check_reflex_recovery_environment.py` to report environment, JAX/MuJoCo/Warp, cache, auth, and checkpoint resolution status without printing tokens.
4. Add non-default Hydra configs:
   - `fullbody/conf_reflex_recovery_smoke.yaml`
   - `fullbody/conf_reflex_recovery_train.yaml`
   - `fullbody/conf_reflex_recovery_eval.yaml`
   - `fullbody/conf_reflex_recovery_ablation.yaml`
5. Add `scripts/audit_myofullbody_model.py` and `musclemimic/research/reflex_recovery/model_audit.py` to export model tables and the toe/action-path reports once the exact model can be loaded locally.
6. Only after the above tests pass, add an environment wrapper or subclass that composes the frozen base action, reflex residual, and recovery residual before `DefaultControl.generate_action`.
7. Only after baseline checkpoint loading and smoke rollout pass, add residual PPO plumbing that freezes the base checkpoint with `jax.lax.stop_gradient`.

Initial implementation assumptions to verify at runtime:

- active MyoFullBody actuator count equals the policy action dimension;
- active control type is `DefaultControl`;
- active control mode is compatible with normalized-action composition;
- active MuJoCo/MJX data exposes a generalized-force field suitable for torque impulses;
- trajectory state contains valid `traj_no` and `subtraj_step_no`;
- contact sensors named `r_foot`, `r_toes`, `l_foot`, and `l_toes` exist in the loaded model;
- muscle sensory quantities are present only if enabled in configuration.
