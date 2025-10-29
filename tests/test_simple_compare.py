import pytest
import mikeio
from datetime import datetime
import modelskill as ms


@pytest.fixture
def fn_mod():
    return "tests/testdata/SW/ts_storm_4.dfs0"


@pytest.fixture
def fn_obs():
    return "tests/testdata/SW/eur_Hm0.dfs0"


def test_compare(fn_obs, fn_mod):
    df_mod = mikeio.open(fn_mod).read(items=0).to_dataframe()
    c = ms.match(ms.PointObservation(fn_obs), ms.PointModelResult(df_mod))
    assert c.n_points == 67
    assert c.time[0] == datetime(2017, 10, 27, 0, 0, 0)
    assert c.time[-1] == datetime(2017, 10, 29, 18, 0, 0)


def test_compare_mod_item(fn_obs, fn_mod):
    c = ms.match(ms.PointObservation(fn_obs), ms.PointModelResult(fn_mod, item=0))

    # not very useful assert, but if you don't provide a model name, you'll get a default one
    assert c.mod_names[0] == "ts_storm_4"


def test_compare_mod_item_2(fn_obs, fn_mod):
    df_mod = mikeio.open(fn_mod).read(items=[0, 1, 2]).to_dataframe()
    c = ms.match(ms.PointObservation(fn_obs), ms.PointModelResult(df_mod, item=0))
    assert c.n_points > 0


def test_compare_fn(fn_obs):
    c = ms.match(ms.PointObservation(fn_obs), ms.PointModelResult(fn_obs))
    assert c.n_points == 95


def test_compare_df(fn_obs, fn_mod):
    df_obs = mikeio.open(fn_obs).read().to_dataframe()
    df_mod = mikeio.open(fn_mod).read(items=0).to_dataframe()
    c = ms.match(ms.PointObservation(df_obs), ms.PointModelResult(df_mod))
    assert c.n_points == 67
    assert c.time[0] == datetime(2017, 10, 27, 0, 0, 0)
    assert c.time[-1] == datetime(2017, 10, 29, 18, 0, 0)


def test_compare_fail(fn_obs, fn_mod):
    df_mod = mikeio.open(fn_mod).read(items=[0, 1, 2]).to_dataframe()
    with pytest.raises(ValueError):
        # multiple items in model df -> ambigous
        ms.match(ms.PointObservation(fn_obs), ms.PointModelResult(df_mod))

    df_obs2, fn_mod2 = df_mod, fn_obs
    with pytest.raises(ValueError):
        # multiple items in obs df -> ambigous
        ms.match(ms.PointObservation(df_obs2), ms.PointModelResult(fn_mod2))


def test_compare_obs_item_pointobs(fn_mod):
    o1 = ms.PointObservation("tests/testdata/SW/eur_Hm0_Quality.dfs0", item=0)

    c = ms.match(o1, ms.PointModelResult(fn_mod, item=0))
    assert c.n_points == 67


def test_force_keyword_args(fn_obs, fn_mod):
    with pytest.raises(TypeError):
        ms.match(fn_obs, fn_mod, 0, 0)
