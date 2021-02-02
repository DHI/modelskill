import pytest
import numpy as np

from mikefm_skill.metrics import mean_absolute_error, nash_sutcliffe_efficiency, root_mean_squared_error


def test_nse_optimal():

    np.random.seed(42)
    obs = np.random.uniform(size=100)

    assert nash_sutcliffe_efficiency(obs, obs) == 1.0

def test_nse_suboptimal():

    obs = np.array([1.0,0.5,0])
    mod = np.array([1.0,0.0,0.5])

    assert nash_sutcliffe_efficiency(obs, mod) == 0.0

def test_rmse():
    obs = np.arange(100)
    mod = obs + 1.0

    rmse = root_mean_squared_error(obs,mod)

    assert rmse == 1.0

def test_mae():
    obs = np.arange(100)
    mod = obs + 1.0

    mae = mean_absolute_error(obs,mod)

    assert mae == 1.0
