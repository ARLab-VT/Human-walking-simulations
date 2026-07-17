import numpy as np

from musclemimic.research.reflex_recovery.comparison import pelvis_tilt_rad


def test_pelvis_tilt_identity_and_quarter_turn():
    qpos = np.zeros((2, 7))
    qpos[0, 3] = 1.0
    qpos[1, 3] = np.cos(np.pi / 4)
    qpos[1, 4] = np.sin(np.pi / 4)
    np.testing.assert_allclose(pelvis_tilt_rad(qpos), [0.0, np.pi / 2])
