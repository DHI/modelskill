import pytest
import mikeio
from datetime import datetime
import pandas as pd
import modelskill as ms


@pytest.fixture
def fn_mod():
    return "tests/testdata/SW/ts_storm_4.dfs0"


@pytest.fixture
def fn_obs():
    return "tests/testdata/SW/eur_Hm0.dfs0"


def test_compare(fn_obs, fn_mod):
    df_mod = mikeio.open(fn_mod).read(items=0).to_dataframe()
    with pytest.warns(UserWarning):
        c = ms.match(fn_obs, df_mod)
    assert c.n_points == 67
    assert c.time[0] == datetime(2017, 10, 27, 0, 0, 0)
    assert c.time[-1] == datetime(2017, 10, 29, 18, 0, 0)


def test_compare_mod_item(fn_obs, fn_mod):
    with pytest.warns(UserWarning):
        c = ms.match(fn_obs, fn_mod, mod_item=0)

    # not very useful assert, but if you don't provide a model name, you'll get a default one
    assert c.mod_names[0] == "ts_storm_4"


def test_compare_mod_item_2(fn_obs, fn_mod):
    df_mod = mikeio.open(fn_mod).read(items=[0, 1, 2]).to_dataframe()
    with pytest.warns(UserWarning):
        c = ms.match(fn_obs, df_mod, mod_item=0)
    assert c.n_points > 0


def test_compare_fn(fn_obs):
    c = ms.match(fn_obs, fn_obs, gtype="point")
    assert c.n_points == 95


def test_compare_df(fn_obs, fn_mod):
    df_obs = mikeio.open(fn_obs).read().to_dataframe()
    df_mod = mikeio.open(fn_mod).read(items=0).to_dataframe()
    with pytest.warns(UserWarning):
        c = ms.match(df_obs, df_mod)
    assert c.n_points == 67
    assert c.time[0] == datetime(2017, 10, 27, 0, 0, 0)
    assert c.time[-1] == datetime(2017, 10, 29, 18, 0, 0)


def test_compare_point_obs(fn_obs, fn_mod):
    obs = ms.PointObservation(fn_obs, name="EPL")
    df_mod = mikeio.open(fn_mod).read(items=0).to_dataframe()
    c = ms.match(obs, df_mod)
    assert c.n_points == 67


def test_compare_fail(fn_obs, fn_mod):
    df_mod = mikeio.open(fn_mod).read(items=[0, 1, 2]).to_dataframe()
    with pytest.raises(ValueError):
        # multiple items in model df -> ambigous
        with pytest.warns(UserWarning):
            ms.match(fn_obs, df_mod)

    df_obs2, fn_mod2 = df_mod, fn_obs
    with pytest.raises(ValueError):
        # multiple items in obs df -> ambigous
        with pytest.warns(UserWarning):
            ms.match(df_obs2, fn_mod2)


def test_compare_obs_item(fn_mod):
    with pytest.warns(UserWarning):
        c = ms.match(
            "tests/testdata/SW/eur_Hm0.dfs0", fn_mod, mod_item=0
        )  # obs file has only 1 item, not necessary to specify obs_item
    assert c.n_points == 67

    with pytest.raises(IndexError):
        with pytest.warns(UserWarning):
            ms.match(
                "tests/testdata/SW/eur_Hm0.dfs0", fn_mod, mod_item=0, obs_item=1
            )  # file has only 1 item

    with pytest.warns(UserWarning):
        c = ms.match(
            "tests/testdata/SW/eur_Hm0_Quality.dfs0", fn_mod, mod_item=0, obs_item=0
        )
    assert c.n_points == 67

    with pytest.raises(ValueError):
        with pytest.warns(UserWarning):
            ms.match(
                "tests/testdata/SW/eur_Hm0_Quality.dfs0", fn_mod
            )  # Obs file has multiple items, but we did not specify one


def test_compare_obs_item_pointobs(fn_mod):
    o1 = ms.PointObservation("tests/testdata/SW/eur_Hm0_Quality.dfs0", item=0)

    c = ms.match(o1, fn_mod, mod_item=0)
    assert c.n_points == 67


def test_compare_obs_item_pointobs_inconsistent_item_error(fn_mod):
    o1 = ms.PointObservation("tests/testdata/SW/eur_Hm0_Quality.dfs0", item=0)

    with pytest.raises(ValueError):
        ms.match(o1, fn_mod, mod_item=0, obs_item=1)  # item=0 != obs_item==1


def test_force_keyword_args(fn_obs, fn_mod):
    with pytest.raises(TypeError):
        ms.match(fn_obs, fn_mod, 0, 0)


def test_matching_pointobservation_with_trackmodelresult_is_not_possible():
    # ignore the data
    tdf = pd.DataFrame(
        {"x": [1, 2], "y": [1, 2], "m1": [0, 0]},
        index=pd.date_range("2017-10-27 13:00:01", periods=2, freq="4s"),
    )
    mr = ms.TrackModelResult(tdf, item="m1", x_item="x", y_item="y")
    pdf = pd.DataFrame(
        data={"level": [0.0, 0.0]},
        index=pd.date_range("2017-10-27 13:00:01", periods=2, freq="4s"),
    )
    obs = ms.PointObservation(pdf, item="level")
    with pytest.raises(TypeError, match="TrackModelResult"):
        ms.match(obs=obs, mod=mr)
