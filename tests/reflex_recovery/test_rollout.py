from pathlib import Path

import numpy as np

from musclemimic.research.reflex_recovery.rollout import load_baseline_rollout, scientific_arrays_equal


def test_load_and_compare_baseline_rollout(tmp_path: Path):
    path = tmp_path / "rollout.npz"
    values = {
        "episode_0_traj_id": 0,
        "episode_0_traj_qpos": np.ones((3, 2)),
        "episode_0_joint_positions": np.ones((2, 2)),
        "episode_0_joint_velocities": np.ones((2, 2)),
        "episode_0_joint_accelerations": np.zeros((2, 2)),
        "episode_0_touch_observations": np.ones((2, 4)),
        "episode_0_policy_actions": np.ones((2, 3)),
        "episode_0_muscle_commands": np.ones((2, 3)),
        "episode_0_muscle_activations": np.ones((2, 3)),
        "episode_0_rewards": np.array([1.0, 2.0]),
        "episode_0_timesteps": np.array([0.0, 0.01]),
        "n_episodes": 1,
        "dt": 0.01,
        "joint_names": np.array(["a", "b"]),
        "env_name": "MyoFullBody",
        "backend": "mujoco",
    }
    np.savez(path, **values)
    rollout = load_baseline_rollout(path)
    assert rollout.num_steps == 2
    assert rollout.total_reward == 3.0
    assert scientific_arrays_equal(rollout, load_baseline_rollout(path))
