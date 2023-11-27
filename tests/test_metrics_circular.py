import itertools
import numpy as np
import pytest

import modelskill.metrics as mtr


# max_error
def test_max_error_circular():
    obs = np.arange(100)
    mod = obs + 1.0

    assert mtr.c_max_error(obs, mod) == 1.0


def test_max_error_circular_symmetry():
    obs = np.array([0, 180, 270, 360])
    mod = np.array([90, 270, 0, -270])
    assert mtr.c_max_error(obs, mod) == 90.0


def test_max_error_circular_zero():
    obs = np.array([0, 45, 90, 135, 180])
    mod = np.array([0, 45, 90, 135, 180])  # Same as obs
    assert mtr.c_max_error(obs, mod) == 0.0


def test_max_error_circular_wrap_around():
    obs = np.array([359, 2, 3, 4])
    mod = np.array([1, 359, 357, 356])
    assert mtr.c_max_error(obs, mod) == 8.0


def test_max_error_circular_opposite():
    obs = np.array([0, 90, 180, 270])
    mod = np.array([180, 270, 0, 90])  # Opposite directions
    assert mtr.c_max_error(obs, mod) == 180.0


def test_max_error_circular_quadrant_boundary():
    obs = np.array([90, 180, 270, 0])
    mod = np.array([89, 181, 271, 359])
    assert mtr.c_max_error(obs, mod) == 1.0


def test_max_error_circular_mixed_directions():
    obs = np.array([10, 20, 150, 340])
    mod = np.array([20, 10, 145, 15])
    assert mtr.c_max_error(obs, mod) == 35.0  # 340 - 15


# mae
def test_mae_circular():
    obs = np.arange(100)
    mod = obs + 1.0

    assert mtr.c_mae(obs, mod) == 1.0


def test_mae_circular_uniform_distribution():
    obs = np.array([0, 90, 180, 270])
    mod = np.array([0, 90, 180, 270])
    assert mtr.c_mae(obs, mod) == 0.0


def test_mae_circular_wrap_around():
    obs = np.array([359, 2, 3, 4])
    mod = np.array([1, 359, 357, 356])
    assert mtr.c_mae(obs, mod) == 4.75


def test_mae_circular_mirror():
    obs = np.array([0, 90, 180, 270])
    mod = np.array([0, -90, 180, -270])  # mirrored across x-axis
    assert mtr.c_mae(obs, mod) == 90.0


def test_mae_circular_opposite_direction():
    obs = np.array([0, 180, 90, 45])
    mod = np.array([180, 0, 270, 225])
    assert mtr.c_mae(obs, mod) == 180.0


def test_mae_circular_perpendicular():
    obs = np.array([0, 0, 0, 0])
    mod = np.array([90, 90, 90, 90])  # Perpendicular to obs
    assert mtr.c_mae(obs, mod) == 90.0


# rmse
def test_rmse_circular():
    obs = np.arange(100)
    mod = obs + 1.0

    assert mtr.c_rmse(obs, mod) == 1.0


def test_rmse_circular_exact_matching():
    obs = np.array([0, 90, 180, 270])
    mod = np.array([360, 450, 540, 630])  # 360 degrees offset
    assert mtr.c_rmse(obs, mod) == 0.0


def test_rmse_circular_quadrant():
    obs = np.array([0, 0, 0, 0])
    mod = np.array([0, 90, 180, 270])  # Covering each quadrant
    assert mtr.c_rmse(obs, mod) > 0.0


def test_rmse_circular_halfway():
    obs = np.array([0, 0, 0, 0])
    mod = np.array([45, 45, 45, 45])  # Halfway between obs and perpendicular
    assert mtr.c_rmse(obs, mod) == 45.0


def test_rmse_circular_nearly_full_circle():
    obs = np.array([0, -1, 0.5, 1])
    mod = np.array([359, 360, 359.5, 360])
    assert mtr.c_rmse(obs, mod) == 1.0


def test_rmse_circular_weighted():
    obs = np.array([0, 180, 90, 270])
    mod = np.array([90, 0, 270, 90])
    weights = np.array([1, 1, 0, 0])
    assert mtr.c_rmse(obs, mod, weights=weights) == pytest.approx(142.3, 0.01)


name2func = {
    "bias": mtr.bias,
    "max_error": mtr.max_error,
    "rmse": mtr.rmse,
    "mae": mtr.mae,
    "urmse": mtr.urmse,
}
name2cfunc = {
    "bias": mtr.c_bias,
    "max_error": mtr.c_max_error,
    "rmse": mtr.c_rmse,
    "mae": mtr.c_mae,
    "urmse": mtr.c_urmse,
}


@pytest.mark.parametrize(
    "metric",
    [
        "bias",
        "max_error",
        "rmse",
        "mae",
        "urmse",
    ],
)
def test_metrics_consistency(metric):
    x = np.array([0, 5, 15, 30])
    y = np.array([1, 2, 9, 21])

    # Standard and circular versions should give same result (for small angles)
    func = name2func[metric]
    v = func(x, y)
    cfunc = name2cfunc[metric]
    vc = (cfunc(x, y) + 180) % 360 - 180
    assert v == pytest.approx(vc, 1e-2)


mtr_funs = [
    mtr.c_bias,
    mtr.c_max_error,
    mtr.c_rmse,
    mtr.c_mae,
    mtr.c_urmse,
]

rotations = [-4, 170, 340]


@pytest.mark.parametrize("func,rot", itertools.product(mtr_funs, rotations))
def test_metrics_consistency_rotated(func, rot):
    x = np.array([0, 5, 15, 30])
    y = np.array([1, 2, 9, 21])
    vc = func(x, y)

    # Should give same result when rotated

    # Rotate x and y by r degrees (0 to 360)
    x2 = (x + rot) % 360
    y2 = (y + rot) % 360
    vc2 = func(x2, y2)
    assert vc == pytest.approx(vc2, 1e-7)

    # Rotate x and y by r degrees (-180 to 180)
    x2 = (x + rot + 180) % 360 - 180
    y2 = (y + rot + 180) % 360 - 180
    vc2 = func(x2, y2)
    assert vc == pytest.approx(vc2, 1e-7)
