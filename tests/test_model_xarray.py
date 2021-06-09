from datetime import datetime
import pytest
import xarray as xr

from fmskill.model import ModelResult
from fmskill.model.abstract import ModelResultInterface
from fmskill.model import XArrayModelResult, XArrayModelResultItem
from fmskill.observation import PointObservation


@pytest.fixture
def ERA5_DutchCoast_nc():
    return r"tests\testdata\SW\ERA5_DutchCoast.nc"


def test_XarrayModelResult_nc(ERA5_DutchCoast_nc):
    mr = ModelResult(ERA5_DutchCoast_nc)

    assert isinstance(mr, XArrayModelResult)
    assert isinstance(mr.ds, xr.Dataset)
    assert len(mr) == 5
    assert len(mr.ds) == 5
    assert mr.name == "ERA5_DutchCoast"
    assert len(mr.item_names) == 5
    assert mr.item_names == ["mwd", "mwp", "mp2", "pp1d", "swh"]


def test_XarrayModelResultItem_selection(ERA5_DutchCoast_nc):
    mr = ModelResult(ERA5_DutchCoast_nc)

    assert isinstance(mr["mwd"], XArrayModelResultItem)
    assert isinstance(mr[0], XArrayModelResultItem)


def test_XarrayModelResultItem(ERA5_DutchCoast_nc):
    mr = ModelResult(ERA5_DutchCoast_nc)
    mri = mr[0]

    assert isinstance(mri.ds, xr.DataArray)  # or mri.da?
    assert len(mri) == 1
    assert len(mri.ds) == 1
    assert mri.name == "ERA5_DutchCoast"
    assert len(mri.item_names) == 1  # or mri.item_name?
    assert mri.item_names == ["mwd"]  # or "mwd"?
