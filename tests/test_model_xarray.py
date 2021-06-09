from datetime import datetime
import pytest
import xarray as xr

from fmskill.model import ModelResult
from fmskill.model.abstract import ModelResultInterface
from fmskill.model import XArrayModelResult, XArrayModelResultItem
from fmskill.observation import PointObservation, TrackObservation


@pytest.fixture
def ERA5_DutchCoast_nc():
    return r"tests\testdata\SW\ERA5_DutchCoast.nc"


@pytest.fixture
def modelresult(ERA5_DutchCoast_nc):
    return ModelResult(ERA5_DutchCoast_nc)


@pytest.fixture
def o1():
    return PointObservation(
        "../tests/testdata/SW/eur_Hm0.dfs0", item=0, x=3.2760, y=51.9990, name="EPL"
    )


@pytest.fixture
def o2():
    return TrackObservation(
        "../tests/testdata/SW/Alti_c2_Dutch.dfs0", item=3, name="c2"
    )


def test_XArrayModelResult_nc(modelresult):
    mr = modelresult

    assert isinstance(mr, XArrayModelResult)
    assert isinstance(mr.ds, xr.Dataset)
    assert len(mr) == 5
    assert len(mr.ds) == 5
    assert mr.name == "ERA5_DutchCoast"
    assert mr.item_names == ["mwd", "mwp", "mp2", "pp1d", "swh"]
    assert mr.start_time == datetime(2017, 10, 27, 0, 0, 0)
    assert mr.end_time == datetime(2017, 10, 29, 18, 0, 0)


def test_XArrayModelResult_ds(ERA5_DutchCoast_nc):
    ds = xr.open_dataset(ERA5_DutchCoast_nc)
    mr = ModelResult(ds)

    assert isinstance(mr, XArrayModelResult)
    assert isinstance(mr.ds, xr.Dataset)
    assert mr.item_names == ["mwd", "mwp", "mp2", "pp1d", "swh"]


def test_XArrayModelResultItem_selection(modelresult):
    mr = modelresult

    assert isinstance(mr["mwd"], XArrayModelResultItem)
    assert isinstance(mr[0], XArrayModelResultItem)


def test_XArrayModelResultItem(modelresult):
    mr = modelresult
    mri = mr[0]

    assert isinstance(mri.ds, xr.DataArray)  # or mri.da?
    assert len(mri) == 1
    assert len(mri.ds) == 1
    assert mri.name == "ERA5_DutchCoast"
    assert mri.item_names == ["mwd"]  # or "mwd"?


def test_XArrayModelResultItem_da(ERA5_DutchCoast_nc):
    da = xr.open_dataset(ERA5_DutchCoast_nc)
    mri = ModelResult(da)

    assert isinstance(mri, XArrayModelResultItem)


def test_ModelResult_extract_point(modelresult, o1):
    mr = modelresult
    pc = mr.extract_observation(o1)  # should this be supported? Find o1 in mr?


def test_ModelResultItem_extract_point(modelresult, o1):
    mr = modelresult
    mri = mr["swh"]
    pc = mri.extract_observation(o1)