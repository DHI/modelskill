import pytest
import numpy as np
from mikeio import Dfs0
from datetime import datetime
import fmskill


@pytest.fixture
def fn_mod():
    return "tests/testdata/SW/ts_storm_4.dfs0"


@pytest.fixture
def fn_obs():
    return "tests/testdata/SW/eur_Hm0.dfs0"


def test_compare(fn_obs, fn_mod):
    df_mod = Dfs0(fn_mod).read(items=0).to_dataframe()
    c = fmskill.compare(fn_obs, df_mod)
    assert c.n_points == 67
    assert c.start == datetime(2017, 10, 27, 0, 0, 0)
    assert c.end == datetime(2017, 10, 29, 18, 0, 0)


def test_compare_fn(fn_obs):
    c = fmskill.compare(fn_obs, fn_obs)
    assert c.n_points == 95


def test_compare_df(fn_obs, fn_mod):
    df_obs = Dfs0(fn_obs).read().to_dataframe()
    df_mod = Dfs0(fn_mod).read(items=0).to_dataframe()
    c = fmskill.compare(df_obs, df_mod)
    assert c.n_points == 67
    assert c.start == datetime(2017, 10, 27, 0, 0, 0)
    assert c.end == datetime(2017, 10, 29, 18, 0, 0)


def test_compare_point_obs(fn_obs, fn_mod):
    obs = fmskill.PointObservation(fn_obs, name="EPL")
    df_mod = Dfs0(fn_mod).read(items=0).to_dataframe()
    c = fmskill.compare(obs, df_mod)
    assert c.n_points == 67


def test_compare_fail(fn_obs, fn_mod):
    df_mod = Dfs0(fn_mod).read(items=[0, 1, 2]).to_dataframe()
    with pytest.raises(ValueError):
        # multiple items in model df -> ambigous
        fmskill.compare(fn_obs, df_mod)

    df_obs2, fn_mod2 = df_mod, fn_obs
    with pytest.raises(ValueError):
        # multiple items in obs df -> ambigous
        fmskill.compare(df_obs2, fn_mod2)


def test_compare_df_residual(fn_obs, fn_mod):
    df_obs = Dfs0(fn_obs).read().to_dataframe()
    df_mod = Dfs0(fn_mod).read(items=0).to_dataframe()
    c = fmskill.compare(df_obs, df_mod)
    assert c.residual.shape[0] <= df_obs.shape[0]