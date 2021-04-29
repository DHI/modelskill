import pytest
import pandas as pd
import xarray as xr
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
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_2.dfsu"
    mr2 = ModelResult(fn, name="SW_2")
    mr = ModelResultCollection([mr1, mr2])

    mr.add_observation(o1, item=0)
    mr.add_observation(o2, item=0)
    mr.add_observation(o3, item=0)
    return mr.extract()


def test_spatial_skill(cc1):
    ss = cc1.spatial_skill()
    assert isinstance(ss.ds, xr.Dataset)
    assert len(ss.x) == 5
    assert len(ss.y) == 5


def test_plot(cc1):
    ss = cc1.spatial_skill(metrics=["rmse", "bias"])
    ss.plot("bias")
