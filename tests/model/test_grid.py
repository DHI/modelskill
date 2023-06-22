from datetime import datetime
import pytest
import xarray as xr
import pandas as pd


import modelskill
from modelskill import ModelResult
from modelskill.model import (
    GridModelResult,
    PointModelResult,
    TrackModelResult,
)
from modelskill.observation import PointObservation, TrackObservation


@pytest.fixture
def ERA5_DutchCoast_nc():
    return "tests/testdata/SW/ERA5_DutchCoast.nc"


@pytest.fixture
def mr_ERA5_pp1d(ERA5_DutchCoast_nc):
    return ModelResult(ERA5_DutchCoast_nc, name="ERA5_DutchCoast", item="pp1d")


@pytest.fixture
def mr_ERA5_swh(ERA5_DutchCoast_nc):
    return ModelResult(ERA5_DutchCoast_nc, name="ERA5_DutchCoast", item="swh")


@pytest.fixture
def mf_modelresult():
    fn = "tests/testdata/SW/CMEMS_DutchCoast_*.nc"
    return ModelResult(fn, item="VHM0", name="CMEMS")


@pytest.fixture
def pointobs_epl_hm0():
    return PointObservation(
        "tests/testdata/SW/eur_Hm0.dfs0", item=0, x=3.2760, y=51.9990, name="EPL"
    )


@pytest.fixture
def trackobs_c2_hm0():
    return TrackObservation("tests/testdata/SW/Alti_c2_Dutch.dfs0", item=3, name="c2")


def test_grid_from_nc(mr_ERA5_pp1d):
    mr = mr_ERA5_pp1d
    assert mr.name == "ERA5_DutchCoast"
    assert mr.start_time == datetime(2017, 10, 27, 0, 0, 0)
    assert mr.end_time == datetime(2017, 10, 29, 18, 0, 0)


def test_grid_from_DataArray(ERA5_DutchCoast_nc):
    ds = xr.open_dataset(ERA5_DutchCoast_nc)
    mr = ModelResult(ds["swh"])

    assert isinstance(mr, GridModelResult)
    assert isinstance(mr.data, xr.Dataset)

    # TODO get quantity info from nc
    # assert mr.quantity.name == "Significant Wave Height"
    assert mr.quantity.name == "Undefined"


def test_dataset_with_missing_coordinates(ERA5_DutchCoast_nc):
    ds = xr.open_dataset(ERA5_DutchCoast_nc)
    ds = ds.drop_vars(["longitude"])  # remove one of the coordinates

    with pytest.raises(ValueError, match="gtype"):
        ModelResult(ds["swh"])


def test_grid_from_da(ERA5_DutchCoast_nc):
    ds = xr.open_dataset(ERA5_DutchCoast_nc)
    da = ds["swh"]
    mr = ModelResult(da)

    assert isinstance(mr, GridModelResult)
    # assert not mr.filename


def test_grid_from_multifile(mf_modelresult):
    mr = mf_modelresult

    assert mr.name == "CMEMS"
    assert mr.start_time == datetime(2017, 10, 28, 0, 0, 0)
    assert mr.end_time == datetime(2017, 10, 29, 18, 0, 0)


# should be supported
def test_grid_name(ERA5_DutchCoast_nc):
    mri1 = ModelResult(ERA5_DutchCoast_nc, item="pp1d")
    assert isinstance(mri1, GridModelResult)

    mri2 = ModelResult(ERA5_DutchCoast_nc, item=3)
    assert isinstance(mri2, GridModelResult)

    assert mri1.name == mri2.name


# def test_grid_itemInfo(ERA5_DutchCoast_nc):
#     mri1 = ModelResult(ERA5_DutchCoast_nc, item="pp1d")
#     assert mri1.itemInfo == mikeio.ItemInfo(mikeio.EUMType.Undefined)

#     itemInfo = mikeio.EUMType.Wave_period
#     mri3 = ModelResult(ERA5_DutchCoast_nc, item="pp1d", itemInfo=itemInfo)
#     mri3.itemInfo == mikeio.ItemInfo(mikeio.EUMType.Wave_period)

#     itemInfo = mikeio.ItemInfo("Peak period", mikeio.EUMType.Wave_period)
#     mri3 = ModelResult(ERA5_DutchCoast_nc, item="pp1d", itemInfo=itemInfo)
#     mri3.itemInfo == mikeio.ItemInfo("Peak period", mikeio.EUMType.Wave_period)


def test_grid_extract_point(mr_ERA5_swh, pointobs_epl_hm0):
    pmr = mr_ERA5_swh.extract(pointobs_epl_hm0)
    df = pmr.data

    assert isinstance(pmr, PointModelResult)
    assert pmr.start_time == datetime(2017, 10, 27, 0, 0, 0)
    assert pmr.end_time == datetime(2017, 10, 29, 18, 0, 0)
    assert len(df.dropna()) == 67
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 1
    assert pytest.approx(df.iloc[0, 0]) == 0.875528


def test_grid_validate_point(mf_modelresult, pointobs_epl_hm0):
    mr = mf_modelresult

    ok = mr._validate_start_end(pointobs_epl_hm0)
    assert ok


def test_grid_extract_point_xoutside(mr_ERA5_pp1d, pointobs_epl_hm0):
    mri = mr_ERA5_pp1d
    pointobs_epl_hm0.x = -50
    with pytest.raises(ValueError, match="outside"):
        mri.extract(pointobs_epl_hm0)


def test_grid_extract_point_toutside(ERA5_DutchCoast_nc, pointobs_epl_hm0):
    ds = xr.open_dataset(ERA5_DutchCoast_nc)
    da = ds["swh"].isel(time=slice(10, 15))
    da["time"] = pd.Timedelta("365D") + da.time
    mr = ModelResult(da)
    with pytest.warns(UserWarning, match="outside"):
        mr.extract(pointobs_epl_hm0)


@pytest.mark.skip(
    reason="validation not possible at the moment, allow item mapping for ModelResult and Observation and match on item name?"
)
def test_grid_extract_point_wrongitem(mr_ERA5_pp1d, pointobs_epl_hm0):
    mri = mr_ERA5_pp1d
    pc = mri.extract(pointobs_epl_hm0)
    assert pc is None


def test_grid_extract_track(mr_ERA5_pp1d, trackobs_c2_hm0):
    mri = mr_ERA5_pp1d
    tmr = mri.extract(trackobs_c2_hm0)
    df = tmr.data

    assert isinstance(tmr, TrackModelResult)
    assert tmr.start_time.replace(microsecond=0) == datetime(2017, 10, 27, 12, 52, 52)
    assert tmr.end_time.replace(microsecond=0) == datetime(2017, 10, 29, 12, 51, 28)
    assert len(df.dropna()) == 99


# TODO: move to test_connector.py
# TODO this test seems to be broken, comparing peak period with significant wave height 🤨
def test_xarray_connector(mr_ERA5_pp1d, pointobs_epl_hm0, trackobs_c2_hm0):
    con = modelskill.Connector([pointobs_epl_hm0, trackobs_c2_hm0], mr_ERA5_pp1d)
    assert len(con) == 2
    assert con.n_models == 1

    cc = con.extract()
    assert len(cc) == 2
