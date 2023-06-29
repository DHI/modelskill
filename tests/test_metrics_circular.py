import numpy as np

import modelskill.metrics as mtr

def test_bias_circular():
    obs = np.arange(100)
    mod = obs + 1.0

    assert mtr.bias(obs, mod, circular=True) == 1.0

def test_max_error_circular():
    obs = np.arange(100)
    mod = obs + 1.0

    assert mtr.max_error(obs, mod, circular=True) == 1.0

def test_mean_circular():
    obs = np.arange(100)
    
    assert mtr._mean(obs, circular=True) == 0.5

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

    assert mtr.urmse(obs, mod, circular=True) == 1.0