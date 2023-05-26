import pytest
import pandas as pd
import xarray as xr
from modelskill import (
    ModelResult,
    PointObservation,
    TrackObservation,
    Connector,
)


@pytest.fixture
def cc1():
    fn = "tests/testdata/NorthSeaHD_and_windspeed.dfsu"
    mr = ModelResult(fn, item=0, name="HD")
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    df = pd.read_csv(fn, index_col=0, parse_dates=True)
    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        o1 = TrackObservation(df, item=2, name="alti")

    con = Connector(o1, mr)
    return con.extract()


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
    mr1 = ModelResult(fn, item=0, name="SW_1")
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    mr2 = ModelResult(fn, item=0, name="SW_2")

    con = Connector([o1, o2, o3], [mr1, mr2])
    return con.extract()


def test_spatial_skill(cc1):
    ss = cc1.spatial_skill()
    assert isinstance(ss.ds, xr.Dataset)
    assert len(ss.x) == 5
    assert len(ss.y) == 5
    assert len(ss.mod_names) == 0
    assert len(ss.obs_names) == 0
    df = ss.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "Coordinates:" in repr(ss)
    assert ss.coords is not None
    assert ss.n is not None


def test_spatial_skill_multi_model(cc2):
    ss = cc2.spatial_skill(bins=3, metrics=["rmse", "bias"])
    assert len(ss.x) == 3
    assert len(ss.y) == 3
    assert len(ss.mod_names) == 2
    assert len(ss.obs_names) == 3
    assert len(ss.field_names) == 3


def test_spatial_skill_plot(cc1):
    ss = cc1.spatial_skill(metrics=["rmse", "bias"])
    ss.plot("bias")


def test_spatial_skill_plot_multi_model(cc2):
    ss = cc2.spatial_skill(by=["model"])
    ss.plot("bias")

    ss.plot("rmse", model="SW_1")

    with pytest.raises(ValueError):
        ss.plot("bad_metric")

    with pytest.raises(ValueError):
        ss.plot("rmse", model="bad_model")
