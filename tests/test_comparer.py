import numpy as np
import pytest
import pandas as pd
import xarray as xr
import fmskill.comparison


def _get_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Observation": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x": [10.1, 10.2, 10.3, 10.4, 10.5],
            "y": [55.1, 55.2, 55.3, 55.4, 55.5],
            "m1": [1.5, 2.4, 3.6, 4.9, 5.6],
            "m2": [1.1, 2.2, 3.1, 4.2, 5.1],
        },
        index=pd.date_range("2019-01-01", periods=5, freq="D"),
    )
    df.index.name = "time"
    return df


def _set_attrs(data: xr.Dataset) -> xr.Dataset:
    data.attrs["variable_name"] = "fake var"
    data["x"].attrs["kind"] = "position"
    data["y"].attrs["kind"] = "position"
    data["Observation"].attrs["kind"] = "observation"
    data["m1"].attrs["kind"] = "model"
    data["m2"].attrs["kind"] = "model"
    return data


@pytest.fixture
def pc() -> fmskill.comparison.Comparer:
    """A comparer with fake point data"""
    x, y = 10.0, 55.0
    df = _get_df().drop(columns=["x", "y"])
    raw_data = {"m1": df[["m1"]], "m2": df[["m2"]]}

    data = df.to_xarray()
    data.attrs["gtype"] = "point"
    data.attrs["name"] = "fake point obs"
    data["x"] = x
    data["y"] = y
    data = _set_attrs(data)
    return fmskill.comparison.Comparer(matched_data=data, raw_mod_data=raw_data)


@pytest.fixture
def tc() -> fmskill.comparison.Comparer:
    """A comparer with fake track data"""
    df = _get_df()
    raw_data = {"m1": df[["x", "y", "m1"]], "m2": df[["x", "y", "m2"]]}

    data = df.to_xarray()
    data.attrs["gtype"] = "track"
    data.attrs["name"] = "fake track obs"
    data = _set_attrs(data)

    return fmskill.comparison.Comparer(matched_data=data, raw_mod_data=raw_data)


def test_pc_properties(pc):
    assert pc.n_models == 2
    assert pc.n_points == 5
    assert pc.gtype == "point"
    assert pc.x == 10.0
    assert pc.y == 55.0
    assert pc.name == "fake point obs"
    assert pc.variable_name == "fake var"
    assert pc.start == pd.Timestamp("2019-01-01")
    assert pc.end == pd.Timestamp("2019-01-05")
    assert pc.mod_names == ["m1", "m2"]


def test_pc_sel_time(pc):
    pc2 = pc.sel(time=slice("2019-01-03", "2019-01-04"))
    assert pc2.n_points == 2
    assert pc2.data.Observation.values.tolist() == [3.0, 4.0]


def test_pc_sel_time_empty(pc):
    pc2 = pc.sel(time=slice("2019-01-06", "2019-01-07"))
    assert pc2.n_points == 0


def test_pc_sel_model(pc):
    pc2 = pc.sel(model="m2")
    assert pc2.n_points == 5
    assert pc2.n_models == 1
    assert np.all(pc2.data.m2 == pc.data.m2)


def test_tc_sel_area(tc):
    bbox = [9.9, 54.9, 10.25, 55.25]
    tc2 = tc.sel(area=bbox)
    assert tc2.n_points == 2
    assert tc2.data.Observation.values.tolist() == [1.0, 2.0]



def test_pc_where(pc):
    pc2 = pc.where(pc.data.Observation > 2.5)
    assert pc2.n_points == 3
    assert pc2.data.Observation.values.tolist() == [3.0, 4.0, 5.0]


def test_pc_where_empty(pc):
    pc2 = pc.where(pc.data.Observation > 10.0)
    assert pc2.n_points == 0


def test_pc_where_derived(pc):
    pc.data["derived"] = pc.data.m1 + pc.data.m2
    pc2 = pc.where(pc.data.derived > 5.0)
    assert pc2.n_points == 3
    assert pc2.data.Observation.values.tolist() == [3.0, 4.0, 5.0]


def test_tc_where_derived(tc):
    x, y = 10.0, 55.0
    dist = np.sqrt((tc.data.x - x) ** 2 + (tc.data.y - y) ** 2)
    # dist = sqrt(2)*[0.1, 0.2, 0.3, 0.4, 0.5]
    tc2 = tc.where(dist > 0.4)
    assert tc2.n_points == 3
    assert tc2.data.Observation.values.tolist() == [3.0, 4.0, 5.0]
