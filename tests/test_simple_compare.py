import pytest
import numpy as np
from mikeio import Dfs0
import fmskill


@pytest.fixture
def fn_mod():
    return "tests/testdata/SW/ts_storm_4.dfs0"


@pytest.fixture
def fn_obs():
    return "tests/testdata/SW/eur_Hm0.dfs0"


def test_compare(fn_mod, fn_obs):
    df_mod = Dfs0(fn_mod).read(items=0).to_dataframe()
    c = fmskill.compare(df_mod, fn_obs)
    assert c.n_points == 66


def test_compare_fn(fn_obs):
    c = fmskill.compare(fn_obs, fn_obs)
    assert c.n_points == 95


def test_compare_df(fn_mod, fn_obs):
    df_mod = Dfs0(fn_mod).read(items=0).to_dataframe()
    df_obs = Dfs0(fn_obs).read().to_dataframe()
    c = fmskill.compare(df_mod, df_obs)
    assert c.n_points == 66


def test_compare_point_obs(fn_mod, fn_obs):
    df_mod = Dfs0(fn_mod).read(items=0).to_dataframe()
    obs = fmskill.PointObservation(fn_obs, name="EPL")
    c = fmskill.compare(df_mod, obs)
    assert c.n_points == 66


def test_compare_fail(fn_mod, fn_obs):
    df_mod = Dfs0(fn_mod).read(items=[0, 1, 2]).to_dataframe()
    with pytest.raises(ValueError):
        # multiple items in model df -> ambigous
        c = fmskill.compare(df_mod, fn_obs)

    df_obs2, fn_mod2 = df_mod, fn_obs
    with pytest.raises(ValueError):
        # multiple items in obs df -> ambigous
        c = fmskill.compare(fn_mod2, df_obs2)
