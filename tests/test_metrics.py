from typing import Callable
import pytest
import numpy as np
import mikeio
import pandas as pd
import modelskill.metrics as mtr


@pytest.fixture
def obs_series() -> pd.Series:
    return (
        mikeio.read("./tests/testdata/PR_test_data.dfs0", items=0)
        .to_dataframe()
        .iloc[:, 0]
    )


@pytest.fixture
def mod_series() -> pd.Series:
    return (
        mikeio.read("./tests/testdata/PR_test_data.dfs0", items=1)
        .to_dataframe()
        .iloc[:, 0]
    )


def test_nse_optimal():
    np.random.seed(42)
    obs = np.random.uniform(size=100)

    assert mtr.nash_sutcliffe_efficiency(obs, obs) == 1.0


def test_kge_optimal():
    np.random.seed(42)
    obs = np.random.uniform(size=100)

    assert mtr.kling_gupta_efficiency(obs, obs) == 1.0


def test_kge_suboptimal():
    obs = np.array([1.0, 0.5, 0])
    mod = np.array([1.0, 0.0, 0.5])

    assert 0.0 < mtr.kling_gupta_efficiency(obs, mod) < 1.0


def test_kge_no_variation_in_obs_returns_nan():
    obs = np.ones(10)
    np.random.seed(42)
    mod = np.random.uniform(size=10)

    assert np.isnan(mtr.kling_gupta_efficiency(obs, mod))


def test_kge_bad():
    np.random.seed(42)
    obs = np.random.normal(loc=10.0, scale=1.0, size=1000)
    mod = np.random.normal(scale=0.1, size=1000)

    assert mtr.kling_gupta_efficiency(obs, mod) < 0.0


def test_kge_climatology_model():
    """Predicting the mean value results in a KGE=-0.41

    Knoben et al, 2019, Hydrol. Earth Syst. Sci., 23, 4323-4331, 2019
    https://doi.org/10.5194/hess-23-4323-2019
    """
    np.random.seed(42)
    obs = np.random.normal(loc=10.0, scale=1.0, size=100)
    mod = np.full_like(obs, fill_value=obs.mean())

    assert mtr.kling_gupta_efficiency(obs, mod) == pytest.approx(-0.41, abs=1e-2)


def test_nse_suboptimal():
    obs = np.array([1.0, 0.5, 0])
    mod = np.array([1.0, 0.0, 0.5])

    assert mtr.nash_sutcliffe_efficiency(obs, mod) == 0.0


def test_mef_suboptimal():
    obs = np.array([1.0, 0.5, 0])
    mod = np.array([1.0, 0.0, 0.5])

    assert mtr.model_efficiency_factor(obs, mod) > 0.0
    assert mtr.model_efficiency_factor(obs, mod) == (
        1 - np.sqrt(mtr.nash_sutcliffe_efficiency(obs, mod))
    )


def test_bias():
    obs = np.arange(100)
    mod = obs + 1.0

    assert mtr.bias(obs, mod) == 1.0


def test_rmse():
    obs = np.arange(100)
    mod = obs + 1.0

    rmse = mtr.root_mean_squared_error(obs, mod)
    assert rmse == 1.0

    rmse = mtr.root_mean_squared_error(obs, mod, weights=obs)
    assert rmse == 1.0

    rmse = mtr.root_mean_squared_error(obs, mod, unbiased=True)
    assert rmse == 0.0


def test_mae():
    obs = np.arange(100)
    mod = obs + 1.0

    mae = mtr.mean_absolute_error(obs, mod)

    assert mae == 1.0


def test_corrcoef():
    obs = np.arange(100)
    mod = obs + 1.0

    r = mtr.corrcoef(obs, mod)
    assert -1.0 <= r <= 1.0

    r = mtr.corrcoef(obs, mod, weights=obs)
    assert -1.0 <= r <= 1.0


def test_scatter_index():
    obs = np.arange(100)
    mod = obs + 1.0

    si = mtr.scatter_index(obs, mod)

    assert si >= 0.0


def test_r2():
    obs = np.arange(100)
    mod = obs + 1.0

    res = mtr.r2(obs, mod)
    assert np.isscalar(res)
    assert 0.0 <= res <= 1.0


def test_mape():
    obs = np.arange(1, 100)
    mod = obs + 1.0

    res = mtr.mean_absolute_percentage_error(obs, mod)
    assert np.isscalar(res)
    assert 0.0 <= res <= 100.0

    obs = np.ones(10)
    obs[5] = 0.0  # MAPE does not like zeros
    mod = obs + 1.0

    with pytest.warns(
        UserWarning,
        match="Observation is zero, consider to use another metric than MAPE",
    ):
        res = mtr.mean_absolute_percentage_error(obs, mod)

    assert np.isnan(res)


def test_max_error():
    obs = np.array([1.0, 0.5, 0])
    mod = np.array([1.0, 0.0, 0.5])

    assert mtr.max_error(obs, mod) == 0.5


def test_willmott():
    obs = np.array([1.0, 0.5, 0])  # mean 0.5
    mod = np.array([1.0, 0.0, 0.5])  # mean 0.5

    assert mtr.willmott(obs, mod) == pytest.approx(1 - 0.5 / 1.5)


def test_ev():
    obs = np.arange(100)
    mod = obs + 1.0
    ev = mtr.ev(obs, mod)

    assert ev == 1.0


def test_pr(obs_series: pd.Series, mod_series: pd.Series) -> None:
    # Obs needs to be a series as the mode of the time index is used.
    # Will use the same data for a real test of ev
    obs = obs_series
    mod = mod_series

    pr = mtr.pr(obs, mod)

    assert pr == pytest.approx(0.889999947851914)


def test_pr_2(obs_series, mod_series):
    # Obs needs to be a series as the mode of the time index is used.
    # Will use the same data for a real test of ev
    obs = obs_series
    mod = mod_series

    pr = mtr.pr(obs, mod, AAP=8, inter_event_level=0.2)

    assert pr == pytest.approx(0.947499960537655)


def test_metric_has_dimension():
    # the following metrics are dimensionless

    assert not mtr.metric_has_units("nse")
    assert not mtr.metric_has_units(mtr.nash_sutcliffe_efficiency)
    assert not mtr.metric_has_units("kge")
    assert not mtr.metric_has_units("r2")

    # while these metrics are in units of the observations
    assert mtr.metric_has_units("mae")
    assert mtr.metric_has_units("bias")
    assert mtr.metric_has_units("rmse")
    assert mtr.metric_has_units(mtr.rmse)

    with pytest.raises(ValueError):
        mtr.metric_has_units("unknown")


def test_add_metric_is_not_a_valid_metric():
    assert not mtr.is_valid_metric("add_metric")
    assert mtr.is_valid_metric("nse")


def test_get_metric():
    rmse = mtr.get_metric("rmse")
    assert isinstance(rmse, Callable)


def test_rmse_small_is_best() -> None:
    assert mtr.rmse.best == "-"
    assert mtr.small_is_best("rmse")


def test_rmse_has_units() -> None:
    assert mtr.rmse.has_units
    assert mtr.metric_has_units("rmse")


def test_r2_large_is_best() -> None:
    assert mtr.r2.best == "+"
    assert mtr.large_is_best("r2")


def test_r2_has_no_units() -> None:
    assert not mtr.r2.has_units
    assert not mtr.metric_has_units("r2")
