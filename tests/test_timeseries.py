import pytest
import numpy as np
import pandas as pd
import xarray as xr
import modelskill as ms
from modelskill.timeseries import TimeSeries
from modelskill.types import GeometryType


@pytest.fixture
def ds_point():
    # create xr dataset with x, y, time and some data
    x = 0
    y = 3
    time = pd.date_range("2000-01-01", periods=3)
    data = np.random.rand(3)
    ds = xr.Dataset(
        {"dataitem": (["time"], data)},
        coords={"time": time},
    )
    ds.coords["x"] = x
    ds.coords["y"] = y
    ds.attrs["gtype"] = str(GeometryType.POINT)
    ds["dataitem"].attrs["kind"] = "observation"
    return ds


@pytest.fixture
def ds_track():
    # create xr dataset with x, y, time and some data
    x = [0, 1, 2]
    y = [3, 4, 5]
    time = pd.date_range("2000-01-01", periods=3)
    data = np.random.rand(3)
    ds = xr.Dataset(
        {"dataitem": (["time"], data)},
        coords={"time": time, "x": (["time"], x), "y": (["time"], y)},
    )
    ds.attrs["gtype"] = str(GeometryType.TRACK)
    ds["dataitem"].attrs["kind"] = "model"
    return ds


def test_timeseries_point_init(ds_point):
    # test that TimeSeries can be initialized from xr.Dataset
    ts = TimeSeries(ds_point)
    assert isinstance(ts, TimeSeries)
    assert isinstance(ts.data, xr.Dataset)
    assert tuple(ts.data.dims) == ("time",)


def test_timeseries_track_init(ds_track):
    # test that TimeSeries can be initialized from xr.Dataset
    ts = TimeSeries(ds_track)
    assert isinstance(ts, TimeSeries)
    assert isinstance(ts.data, xr.Dataset)
    assert tuple(ts.data.dims) == ("time",)


def test_timeseries_validation_fails_gtype(ds_point):
    ds_point.attrs["gtype"] = "GRID"
    with pytest.raises(AssertionError, match="attribute 'gtype' must be"):
        TimeSeries(ds_point)

    ds_point.attrs["gtype"] = "point"  # lower case okay
    ts = TimeSeries(ds_point)
    assert isinstance(ts, TimeSeries)
    assert ts.data.attrs["gtype"] == str(GeometryType.POINT)

    ds_point.attrs["gtype"] = "POINT"  # upper case not okay
    with pytest.raises(AssertionError, match="attribute 'gtype' must be"):
        TimeSeries(ds_point)


def test_timeseries_validation_fails_kind(ds_point):
    ds_point["dataitem"].attrs["kind"] = "MODEL"
    with pytest.raises(ValueError, match="kind attribute"):
        TimeSeries(ds_point)

    ds_point["dataitem"].attrs["kind"] = "aux"
    with pytest.raises(ValueError, match="kind attribute"):
        TimeSeries(ds_point)


def test_timeseries_validation_fails_xy(ds_point):
    ds_without_x = ds_point.drop("x")
    with pytest.raises(AssertionError, match="data must have an x-coordinate"):
        TimeSeries(ds_without_x)

    # ds_point.coords["x"] = 0
    ds_without_y = ds_point.drop("y")
    with pytest.raises(AssertionError, match="data must have a y-coordinate"):
        TimeSeries(ds_without_y)


def test_timeseries_point_properties(ds_point):
    ts = TimeSeries(ds_point)
    assert ts.name == "dataitem"
    assert ts.x == 0
    assert ts.y == 3
    assert list(ts.time) == list(pd.date_range("2000-01-01", periods=3))
    assert ts.start_time == pd.Timestamp("2000-01-01")
    assert ts.end_time == pd.Timestamp("2000-01-03")
    assert ts.n_points == 3
    assert len(ts) == 3
    assert len(ts.color) == 7


def test_timeseries_track_properties(ds_track):
    ts = TimeSeries(ds_track)
    assert ts.name == "dataitem"
    assert list(ts.x) == [0, 1, 2]
    assert list(ts.y) == [3, 4, 5]
    assert list(ts.time) == list(pd.date_range("2000-01-01", periods=3))
    assert ts.start_time == pd.Timestamp("2000-01-01")
    assert ts.end_time == pd.Timestamp("2000-01-03")
    assert ts.n_points == 3
    assert len(ts) == 3
    assert len(ts.color) == 7


def test_timeseries_set_name(ds_track):
    ts = TimeSeries(ds_track)
    assert ts.name == "dataitem"
    ts.name = "newname"
    assert ts.name == "newname"

    with pytest.raises(AssertionError, match="must be a string"):
        ts.name = 1

    with pytest.raises(AssertionError, match="must be a string"):
        ts.name = None

    with pytest.raises(AssertionError, match="reserved"):
        ts.name = "x"


def test_timeseries_set_quantity(ds_track):
    ts = TimeSeries(ds_track)
    assert isinstance(ts.quantity, ms.Quantity)
    assert ts.quantity == ms.Quantity.undefined()

    ts.quantity = ms.Quantity("water level", "m")
    assert ts.quantity == ms.Quantity("water level", "m")

    with pytest.raises(AssertionError, match="must be a Quantity"):
        ts.quantity = 1


def test_timeseries_set_color(ds_track):
    ts = TimeSeries(ds_track)
    orig_color = ts.color
    assert isinstance(ts.color, str)

    ts.color = "red"
    assert ts.color == "red"

    ts.color = None
    assert ts.color == orig_color

    ts.color = "0.6"
    assert ts.color == "0.6"

    with pytest.raises(ValueError, match="color"):
        ts.color = "fakeblue"

    with pytest.raises(ValueError, match="color"):
        ts.color = 1


def test_timeseries_point_set_xy(ds_point):
    ts = TimeSeries(ds_point)
    assert ts.x == 0
    assert ts.y == 3

    ts.x = 1
    ts.y = 2
    assert ts.x == 1
    assert ts.y == 2

    # with pytest.raises(AssertionError, match="must be a float"):
    #     ts.x = "1"

    # with pytest.raises(AssertionError, match="must be a float"):
    #     ts.y = None


def test_timeseries_track_set_xy(ds_track):
    ts = TimeSeries(ds_track)
    assert list(ts.x) == [0, 1, 2]
    assert list(ts.y) == [3, 4, 5]

    ts.x = [1, 2, 3]
    ts.y = [4, 5, 6]
    assert list(ts.x) == [1, 2, 3]
    assert list(ts.y) == [4, 5, 6]

    # with pytest.raises(AssertionError):
    #     ts.x = [8, 9] # wrong length


def test_timeseries_point_to_dataframe(ds_point):
    ts = TimeSeries(ds_point)
    df = ts.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert tuple(df.columns) == ("dataitem",)
    assert tuple(df.index.names) == ("time",)
    assert tuple(df.index) == tuple(ts.time)
    assert tuple(df.dataitem) == tuple(ts.data.dataitem)
    assert len(df) == 3


def test_timeseries_track_to_dataframe(ds_track):
    ts = TimeSeries(ds_track)
    df = ts.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert tuple(df.columns) == ("x", "y", "dataitem")
    assert tuple(df.index.names) == ("time",)
    assert tuple(df.index) == tuple(ts.time)
    assert tuple(df.dataitem) == tuple(ts.data.dataitem)
    assert len(df) == 3
