import jax
import jax.numpy as jnp
import numpy as np

from musclemimic.research.reflex_recovery.muscle_groups import (
    MuscleGroup,
    distribute_group_action,
    distribution_matrix,
    validate_groups,
    build_lower_body_groups,
    export_group_map,
)


def test_group_distribution_is_valid_and_differentiable():
    names = ("a", "b", "c")
    groups = (
        MuscleGroup("flex", ("a", "b"), (0, 1), (0.25, 0.75), "left", "flexion"),
        MuscleGroup("extend", ("c",), (2,), (1.0,), "right", "extension"),
    )
    validate_groups(groups, names)
    matrix = distribution_matrix(groups, 3)
    action = jnp.array([[2.0, -1.0]])
    output = distribute_group_action(action, matrix)
    np.testing.assert_allclose(output, [[0.5, 1.5, -1.0]])
    gradient = jax.grad(lambda x: jnp.sum(distribute_group_action(x, matrix)))(action)
    np.testing.assert_allclose(gradient, [[1.0, 1.0]])


def test_default_lower_body_map_has_16_exclusive_normalized_groups(tmp_path):
    base = (
        "iliacus", "psoas", "sart", "glmax1", "glmax2", "glmax3", "glmed1", "glmed2", "glmed3",
        "glmin1", "glmin2", "glmin3", "piri", "tfl", "addbrev", "addlong", "addmagDist", "addmagIsch",
        "addmagMid", "addmagProx", "grac", "recfem", "vasint", "vaslat", "vasmed", "bflh", "bfsh",
        "semimem", "semiten", "gaslat", "gasmed", "soleus", "tibpost", "perbrev", "perlong", "fdl", "fhl",
        "tibant", "edl", "ehl",
    )
    names = tuple(f"{name}_{side}" for side in ("r", "l") for name in base)
    groups = build_lower_body_groups(names)
    assert len(groups) == 16
    assert sum(len(group.actuator_names) for group in groups) == 80
    assert len({index for group in groups for index in group.actuator_indices}) == 80
    export_group_map(groups, tmp_path)
    assert (tmp_path / "muscle_groups.csv").exists()
    assert (tmp_path / "muscle_groups.md").exists()
