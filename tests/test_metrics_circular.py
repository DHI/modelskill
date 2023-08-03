import numpy as np
import pytest

import modelskill.metrics as mtr

def test_mean_circular():
    obs = np.arange(101)
    
    assert mtr._mean(obs, circular=True) == pytest.approx(50.0)

# bias

def test_bias_circular():
    obs = np.arange(100)
    mod = obs + 1.0

    assert mtr.bias(obs, mod, circular=True) == 1.0


def test_bias_circular_wraparound():
    obs = np.array([0, 90, 180, 270])
    mod = np.array([360, 450, 540, 630])  # Same as obs, but wrapped around
    assert mtr.bias(obs, mod, circular=True) == 0.0

def test_bias_circular_quadrant_boundary():
    obs = np.array([90, 180, 270, 360])
    mod = np.array([180, 270, 360, 450])
    assert mtr.bias(obs, mod, circular=True) == 90.0

def test_bias_circular_symmetry():
    obs = np.array([0, 180, 270, 360])
    mod = np.array([90, 270, 0, -270])
    assert mtr.bias(obs, mod, circular=True) == 90.0

# max_error

def test_max_error_circular():
    obs = np.arange(100)
    mod = obs + 1.0

    assert mtr.max_error(obs, mod, circular=True) == 1.0

def test_max_error_circular_symmetry():
    obs = np.array([0, 180, 270, 360])
    mod = np.array([90, 270, 0, -270])
    assert mtr.max_error(obs, mod, circular=True) == 90.0

def test_max_error_circular_zero():
    obs = np.array([0, 45, 90, 135, 180])
    mod = np.array([0, 45, 90, 135, 180])  # Same as obs
    assert mtr.max_error(obs, mod, circular=True) == 0.0

def test_max_error_circular_wrap_around():
    obs = np.array([359, 2, 3, 4])
    mod = np.array([1, 359, 357, 356])
    assert mtr.max_error(obs, mod, circular=True) == 8.0

def test_max_error_circular_opposite():
    obs = np.array([0, 90, 180, 270])
    mod = np.array([180, 270, 0, 90])  # Opposite directions
    assert mtr.max_error(obs, mod, circular=True) == 180.0

def test_max_error_circular_quadrant_boundary():
    obs = np.array([90, 180, 270, 0])
    mod = np.array([89, 181, 271, 359])
    assert mtr.max_error(obs, mod, circular=True) == 1.0

def test_max_error_circular_mixed_directions():
    obs = np.array([10, 20, 150, 340])
    mod = np.array([20, 10, 145, 15])
    assert mtr.max_error(obs, mod, circular=True) == 35.0 # 340 - 15

# mae

def test_mae_circular():
    obs = np.arange(100)
    mod = obs + 1.0

    assert mtr.mae(obs, mod, circular=True) == 1.0

def test_mae_circular_uniform_distribution():
    obs = np.array([0, 90, 180, 270])
    mod = np.array([0, 90, 180, 270])
    assert mtr.mae(obs, mod, circular=True) == 0.0


def test_mae_circular_wrap_around():
    obs = np.array([359, 2, 3, 4])
    mod = np.array([1, 359, 357, 356])
    assert mtr.mae(obs, mod, circular=True) == 4.75


def test_mae_circular_mirror():
    obs = np.array([0, 90, 180, 270])
    mod = np.array([0, -90, 180, -270])  # mirrored across x-axis
    assert mtr.mae(obs, mod, circular=True) == 90.0


def test_mae_circular_opposite_direction():
    obs = np.array([0, 180, 90, 45])
    mod = np.array([180, 0, 270, 225])
    assert mtr.mae(obs, mod, circular=True) == 180.0

def test_mae_circular_perpendicular():
    obs = np.array([0, 0, 0, 0])
    mod = np.array([90, 90, 90, 90])  # Perpendicular to obs
    assert mtr.mae(obs, mod, circular=True) == 90.0

# mape

def test_mape_perfect_prediction():
    obs = np.array([100, 200, 300, 400])
    model = np.array([100, 200, 300, 400])
    assert mtr.mape(obs, model) == pytest.approx(0)

def test_mape_circular_prediction():
    obs = np.array([100, 200, 300, 400])
    model = np.array([400, 300, 200, 100])
    assert mtr.mape(obs, model) == pytest.approx(114.5833)




# rmse

def test_rmse_circular():
    obs = np.arange(100)
    mod = obs + 1.0

    assert mtr.rmse(obs, mod, circular=True) == 1.0

def test_urmse_circular():
    obs = np.arange(100)
    mod = obs + 1.0

    assert mtr.urmse(obs, mod, circular=True) == 0.0

def test_rmse_circular_exact_matching():
    obs = np.array([0, 90, 180, 270])
    mod = np.array([360, 450, 540, 630])  # 360 degrees offset
    assert mtr.rmse(obs, mod, circular=True) == 0.0

def test_rmse_circular_quadrant():
    obs = np.array([0, 0, 0, 0])
    mod = np.array([0, 90, 180, 270])  # Covering each quadrant
    assert mtr.rmse(obs, mod, circular=True) > 0.0

def test_urmse_circular_constant():
    obs = np.array([0, 0, 0, 0])
    mod = np.array([45, 45, 45, 45])  # constant offset
    assert mtr.urmse(obs, mod, circular=True) == 0.0

def test_rmse_circular_halfway():
    obs = np.array([0, 0, 0, 0])
    mod = np.array([45, 45, 45, 45])  # Halfway between obs and perpendicular
    assert mtr.rmse(obs, mod, circular=True) == 45.0

def test_rmse_circular_nearly_full_circle():
    obs = np.array([0, -1, 0.5, 1])
    mod = np.array([359, 360, 359.5, 360])
    assert mtr.rmse(obs, mod, circular=True) == 1.0

# def test_urmse_circular_scattered():
#     obs = np.array([0, 90, 180, 270])
#     mod = np.array([0, 180, 0, 180])  # Errors are both 0 and 180
#     # URMS error should be less than RMSE because it ignores bias
#     assert mtr.urmse(obs, mod, circular=True) < mtr.rmse(obs, mod, circular=True)

def test_rmse_circular_weighted():
    obs = np.array([0, 180, 90, 270])
    mod = np.array([90, 0, 270, 90])
    weights = np.array([1, 1, 0, 0])
    assert mtr.rmse(obs, mod, weights=weights, circular=True) == pytest.approx(142.3, 0.01)


# std


def test_std_circular_uniform():
    # For a uniform distribution from 0 to 180, 
    # the standard deviation should be about 73.5 degrees
    obs = np.array([0, 60, 120, 180])
    assert mtr._std(obs, circular=True) == pytest.approx(73.5, 0.1)

def test_std_circular_same():
    # For all angles the same, the standard deviation should be 0
    obs = np.array([45, 45, 45, 45])
    assert mtr._std(obs, circular=True) == 0.0

def test_std_circular_close():
    # For all angles close to each other, the standard deviation should be low
    obs = np.array([40, 45, 50, 45])
    assert mtr._std(obs, circular=True) == pytest.approx(3.54, 0.1)

def test_std_circular_half_circle():
    # For angles distributed evenly over half a circle, 
    # the standard deviation should be close to 51.96 degrees
    obs = np.array([0, 45, 90, 135, 180])
    assert mtr._std(obs, circular=True) == pytest.approx(69.14, 0.01)


# corrcoef

# def test_circular_corrcoef_perfect_positive():
#     obs = np.array([0, 90, 180, 270])
#     mod = np.array([0, 90, 180, 270])
#     assert mtr.corrcoef(obs, mod, circular=True) == pytest.approx(1.0)


# def test_circular_corrcoef_perfect_negative():
#     obs = np.array([0, 90, 180, 270])
#     mod = np.array([180, 270, 0, 90])
#     assert mtr.corrcoef(obs, mod, circular=True) == pytest.approx(-1.0)


# def test_circular_corrcoef_circular_shift():
#     obs = np.array([0, 90, 180, 270])
#     mod = np.array([90, 180, 270, 0])  # 90-degree shift
#     assert mtr.corrcoef(obs, mod, circular=True) == pytest.approx(0.0)
