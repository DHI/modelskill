import pytest
import mikeio
from datetime import datetime
import modelskill
from modelskill.observation import PointObservation


@pytest.fixture
def fn_mod():
    return "tests/testdata/SW/ts_storm_4.dfs0"


@pytest.fixture
def fn_obs():
    return "tests/testdata/SW/eur_Hm0.dfs0"


def test_compare(fn_obs, fn_mod):
    df_mod = mikeio.open(fn_mod).read(items=0).to_dataframe()
    c = modelskill.compare(fn_obs, df_mod)
    assert c.n_points == 67
    assert c.start == datetime(2017, 10, 27, 0, 0, 0)
    assert c.end == datetime(2017, 10, 29, 18, 0, 0)


def test_compare_mod_item(fn_obs, fn_mod):
    c = modelskill.compare(fn_obs, fn_mod, mod_item=0)
    dfs = mikeio.open(fn_mod)
    assert c.mod_names[0] == dfs.items[0].name


def test_compare_mod_item_2(fn_obs, fn_mod):
    df_mod = mikeio.open(fn_mod).read(items=[0, 1, 2]).to_dataframe()
    c = modelskill.compare(fn_obs, df_mod, mod_item=0)
    assert c.n_points > 0


def test_compare_fn(fn_obs):
    c = modelskill.compare(fn_obs, fn_obs)
    assert c.n_points == 95


def test_compare_df(fn_obs, fn_mod):
    df_obs = mikeio.open(fn_obs).read().to_dataframe()
    df_mod = mikeio.open(fn_mod).read(items=0).to_dataframe()
    c = modelskill.compare(df_obs, df_mod)
    assert c.n_points == 67
    assert c.start == datetime(2017, 10, 27, 0, 0, 0)
    assert c.end == datetime(2017, 10, 29, 18, 0, 0)


def test_compare_point_obs(fn_obs, fn_mod):
    obs = modelskill.PointObservation(fn_obs, name="EPL")
    df_mod = mikeio.open(fn_mod).read(items=0).to_dataframe()
    c = modelskill.compare(obs, df_mod)
    assert c.n_points == 67


def test_compare_fail(fn_obs, fn_mod):
    df_mod = mikeio.open(fn_mod).read(items=[0, 1, 2]).to_dataframe()
    with pytest.raises(ValueError):
        # multiple items in model df -> ambigous
        modelskill.compare(fn_obs, df_mod)

    df_obs2, fn_mod2 = df_mod, fn_obs
    with pytest.raises(ValueError):
        # multiple items in obs df -> ambigous
        modelskill.compare(df_obs2, fn_mod2)


def test_compare_obs_item(fn_mod):

    c = modelskill.compare(
        "tests/testdata/SW/eur_Hm0.dfs0", fn_mod, mod_item=0
    )  # obs file has only 1 item, not necessary to specify obs_item
    assert c.n_points == 67

    with pytest.raises(ValueError):
        modelskill.compare(
            "tests/testdata/SW/eur_Hm0.dfs0", fn_mod, mod_item=0, obs_item=1
        )  # file has only 1 item

    c = modelskill.compare(
        "tests/testdata/SW/eur_Hm0_Quality.dfs0", fn_mod, mod_item=0, obs_item=0
    )
    assert c.n_points == 67

    with pytest.raises(ValueError):
        modelskill.compare(
            "tests/testdata/SW/eur_Hm0_Quality.dfs0", fn_mod
        )  # Obs file has multiple items, but we did not specify one


def test_compare_obs_item_pointobs(fn_mod):

    o1 = PointObservation("tests/testdata/SW/eur_Hm0_Quality.dfs0", item=0)

    c = modelskill.compare(o1, fn_mod, mod_item=0)
    assert c.n_points == 67


def test_compare_obs_item_pointobs_inconsistent_item_error(fn_mod):

    o1 = PointObservation("tests/testdata/SW/eur_Hm0_Quality.dfs0", item=0)

    with pytest.raises(ValueError):
        modelskill.compare(o1, fn_mod, mod_item=0, obs_item=1)  # item=0 != obs_item==1


def test_force_keyword_args(fn_obs, fn_mod):

    with pytest.raises(TypeError):
        modelskill.compare(fn_obs, fn_mod, 0, 0)
