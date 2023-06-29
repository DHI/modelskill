import numpy as np
import pytest

import modelskill.metrics as mtr

def test_mean_circular():
    obs = np.arange(101)
    
    assert mtr._mean(obs, circular=True) == pytest.approx(50.0)

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
    assert mtr._std(obs, circular=True) == pytest.approx(3.5, 0.1)

def test_bias_circular():
    obs = np.arange(100)
    mod = obs + 1.0

    assert mtr.bias(obs, mod, circular=True) == 1.0

def test_max_error_circular():
    obs = np.arange(100)
    mod = obs + 1.0

    assert mtr.max_error(obs, mod, circular=True) == 1.0

def test_mae_circular():
    obs = np.arange(100)
    mod = obs + 1.0

    assert mtr.mae(obs, mod, circular=True) == 1.0

def test_rmse_circular():
    obs = np.arange(100)
    mod = obs + 1.0

    assert mtr.rmse(obs, mod, circular=True) == 1.0

def test_urmse_circular():
    obs = np.arange(100)
    mod = obs + 1.0

    assert mtr.urmse(obs, mod, circular=True) == 0.0