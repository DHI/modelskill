from datetime import datetime
import pytest
import xarray as xr
import pandas as pd
import numpy as np

from fmskill.model import ModelResult
from fmskill.model.abstract import ModelResultInterface
from fmskill.model import XArrayModelResult, XArrayModelResultItem
from fmskill.observation import PointObservation, TrackObservation
from fmskill.comparison import PointComparer, TrackComparer


@pytest.fixture
def ERA5_DutchCoast_nc():
    return r"tests\testdata\SW\ERA5_DutchCoast.nc"


@pytest.fixture
def modelresult(ERA5_DutchCoast_nc):
    return ModelResult(ERA5_DutchCoast_nc)


@pytest.fixture
def pointobs_epl_hm0():
    return PointObservation(
        "tests/testdata/SW/eur_Hm0.dfs0", item=0, x=3.2760, y=51.9990, name="EPL"
    )


@pytest.fixture
def trackobs_c2_hm0():
    return TrackObservation("tests/testdata/SW/Alti_c2_Dutch.dfs0", item=3, name="c2")


def test_XArrayModelResult_from_nc(modelresult):
    mr = modelresult

    assert isinstance(mr, XArrayModelResult)
    assert isinstance(mr.ds, xr.Dataset)
    assert len(mr) == 5
    assert len(mr.ds) == 5
    assert mr.name == "ERA5_DutchCoast"
    assert mr.item_names == ["mwd", "mwp", "mp2", "pp1d", "swh"]
    assert mr.start_time == datetime(2017, 10, 27, 0, 0, 0)
    assert mr.end_time == datetime(2017, 10, 29, 18, 0, 0)


def test_XArrayModelResult_from_ds(ERA5_DutchCoast_nc):
    ds = xr.open_dataset(ERA5_DutchCoast_nc)
    mr = ModelResult(ds)

    assert isinstance(mr, XArrayModelResult)
    assert isinstance(mr.ds, xr.Dataset)
    assert mr.item_names == ["mwd", "mwp", "mp2", "pp1d", "swh"]


def test_XArrayModelResult_from_da(ERA5_DutchCoast_nc):
    ds = xr.open_dataset(ERA5_DutchCoast_nc)
    da = ds["swh"]
    mr = ModelResult(da)

    assert isinstance(mr, XArrayModelResultItem)


# ToDo
# def test_XArrayModelResult_from_grib
# def test_XArrayModelResult_from_mfdataset
# def test_XArrayModelResult_options


def test_XArrayModelResult_select_item(modelresult):
    mr = modelresult

    assert isinstance(mr["mwd"], XArrayModelResultItem)
    assert isinstance(mr[0], XArrayModelResultItem)


def test_XArrayModelResultItem(modelresult):
    mr = modelresult
    mri = mr[0]

    assert isinstance(mri.ds, xr.DataArray)
    assert len(mri) == 1
    assert len(mri.ds) == 1
    assert mri.name == "ERA5_DutchCoast"
    assert mri.item_names == ["mwd"]  # ToDo: or "mwd"?


def test_XArrayModelResult_extract_point(modelresult, pointobs_epl_hm0):
    mr = modelresult
    pc = mr.extract_observation(
        pointobs_epl_hm0
    )  # ToDo: should this be supported? Find o1 in mr?

    assert isinstance(pc, PointComparer)


def test_XArrayModelResultItem_extract_point(modelresult, pointobs_epl_hm0):
    mr = modelresult
    mri = mr["swh"]
    pc = mri.extract_observation(pointobs_epl_hm0)
    df = pc.df

    assert isinstance(pc, PointComparer)
    assert pc.start == datetime(
        2017, 10, 27, 0, 0, 0
    )  # ToDo: start_time like ModelResult?
    assert pc.end == datetime(2017, 10, 29, 18, 0, 0)
    assert pc.n_models == 1
    assert pc.n_points == 67
    assert pc.n_variables == 1
    assert len(df.dropna()) == 67


def test_XArrayModelResultItem_extract_point_xoutside(modelresult, pointobs_epl_hm0):
    mr = modelresult
    mri = mr["swh"]
    pointobs_epl_hm0.x = -50
    pc = mri.extract_observation(pointobs_epl_hm0)

    assert pc == None


def test_XArrayModelResultItem_extract_point_toutside(
    ERA5_DutchCoast_nc, pointobs_epl_hm0
):
    ds = xr.open_dataset(ERA5_DutchCoast_nc)
    da = ds["swh"].isel(time=slice(10, 15))
    da["time"] = pd.Timedelta("1Y") + da.time
    mr = ModelResult(da)
    pc = mr.extract_observation(pointobs_epl_hm0)

    assert pc == None


def test_XArrayModelResultItem_extract_point_wrongitem(modelresult, pointobs_epl_hm0):
    mr = modelresult
    mri = mr["mwd"]
    pc = mri.extract_observation(pointobs_epl_hm0)

    assert pc == None
    # ToDo: validation not possible at the moment, allow item mapping for ModelResult and Observation and match on item name?


# ToDo: def test_ModelResultItem_extract_point_nanintime():


def test_XArrayModelResultItem_extract_track(modelresult, trackobs_c2_hm0):
    mr = modelresult
    mri = mr["swh"]
    tc = mri.extract_observation(trackobs_c2_hm0)
    df = tc.df

    assert isinstance(tc, TrackComparer)
    assert tc.start.replace(microsecond=0) == datetime(2017, 10, 27, 12, 52, 52)
    assert tc.end.replace(microsecond=0) == datetime(2017, 10, 29, 12, 52, 51)
    assert tc.n_models == 1
    assert tc.n_points == 298
    assert tc.n_variables == 1
    assert len(df.dropna()) == 298


# ToDo: include connector test involving xarray in test_connector?