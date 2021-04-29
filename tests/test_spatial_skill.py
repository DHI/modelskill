import pytest
import pandas as pd
import xarray as xr
from fmskill import ModelResult, TrackObservation


@pytest.fixture
def cc():
    fn = "tests/testdata/NorthSeaHD_and_windspeed.dfsu"
    mr = ModelResult(fn, name="HD")
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    df = pd.read_csv(fn, index_col=0, parse_dates=True)
    o1 = TrackObservation(df, item=2, name="alti")
    mr.add_observation(o1, item=0)
    return mr.extract()


def test_spatial_skill(cc):
    ss = cc.spatial_skill()
    assert isinstance(ss.ds, xr.Dataset)  # core.dataset.
    assert len(ss.x) == 5
    assert len(ss.y) == 5


def test_plot(cc):
    ss = cc.spatial_skill(metrics=["rmse", "bias"])
    ss.plot("bias")
