import pytest
import pandas as pd
from fmskill import (
    ModelResult,
    ModelResultCollection,
    PointObservation,
    TrackObservation,
)


@pytest.fixture
def cc1():
    fn = "tests/testdata/NorthSeaHD_and_windspeed.dfsu"
    mr = ModelResult(fn, name="HD")
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    df = pd.read_csv(fn, index_col=0, parse_dates=True)
    o1 = TrackObservation(df, item=2, name="alti")
    mr.add_observation(o1, item=0)
    return mr.extract()


@pytest.fixture
def o1():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def o2():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def o3():
    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    return TrackObservation(fn, item=3, name="c2")


@pytest.fixture
def cc2(o1, o2, o3):
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    mr1 = ModelResult(fn, name="SW_1")
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    mr2 = ModelResult(fn, name="SW_2")
    mr = ModelResultCollection([mr1, mr2])

    mr.add_observation(o1, item=0)
    mr.add_observation(o2, item=0)
    mr.add_observation(o3, item=0)
    return mr.extract()


def test_skill(cc1):
    s = cc1.skill()
    assert isinstance(s.df, pd.DataFrame)
    assert len(s.mod_names) == 0
    assert len(s.obs_names) == 1
    df = s.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "bias" in repr(s)
    assert s.loc["alti"] is not None


def test_skill_multi_model(cc2):
    s = cc2.skill(metrics=["rmse", "bias"])
    assert isinstance(s.index, pd.MultiIndex)
    assert len(s.mod_names) == 2
    assert len(s.obs_names) == 3
    assert len(s.field_names) == 3


def test_skill_sel(cc2):
    s = cc2.skill(metrics=["rmse", "bias"])
    s2 = s.sel(model="SW_1")
    assert len(s2.mod_names) == 0  # no longer in index
    assert not isinstance(s2.index, pd.MultiIndex)
    assert len(s2) == 3

    s2 = s.sel(model="SW_1", observation=["EPL", "c2"])
    assert len(s2.obs_names) == 2
    assert not isinstance(s2.index, pd.MultiIndex)
    assert len(s2) == 2


def test_skill_sel_query(cc2):
    s = cc2.skill(metrics=["rmse", "bias"])
    s2 = s.sel("rmse>0.2")
    assert len(s2.mod_names) == 2

    s2 = s.sel("rmse>0.2", model="SW_2")
    assert len(s2.mod_names) == 0 # no longer in index


def test_skill_sel_columns(cc2):
    s = cc2.skill(metrics=["rmse", "bias"])
    s2 = s.sel(columns=["n", "rmse"])

    s2 = s.sel(columns="rmse")


def test_skill_sel_fail(cc2):
    s = cc2.skill(metrics=["rmse", "bias"])
    with pytest.raises(KeyError):
        s2 = s.sel(columns=["cc"])

    with pytest.raises(KeyError):
        s2 = s.sel(variable="Hm0")


def test_skill_plot_bar(cc1):
    s = cc1.skill(metrics=["rmse", "bias"])
    s.plot_bar("bias")


def test_skill_plot_bar_multi_model(cc2):
    s = cc2.skill(metrics="rmse")
    s.plot_bar("rmse")


def test_skill_plot_multi_model(cc2):
    s = cc2.skill()
    s.plot_line("bias")

    with pytest.raises(KeyError):
        s.plot_bar("bad_metric")
