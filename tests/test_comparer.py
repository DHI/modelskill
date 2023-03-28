import pytest
import pandas as pd
import fmskill.comparison


@pytest.fixture
def pc():
    x, y = 10.0, 55.0
    df = pd.DataFrame(
        {
            "Observation": [1.0, 2.0, 3.0, 4.0, 5.0],
            "m1": [1.5, 2.4, 3.6, 4.9, 5.6],
            "m2": [1.1, 2.2, 3.1, 4.2, 5.1],
        },
        index=pd.date_range("2019-01-01", periods=5, freq="D"),
    )
    df.index.name = "time"
    raw_data = {"m1": df[["m1"]], "m2": df[["m2"]]}

    data = df.to_xarray()
    data.attrs["gtype"] = "point"
    data["x"] = x
    data["y"] = y
    data.attrs["name"] = "fake obs"
    data.attrs["variable_name"] = "fake var"
    data["x"].attrs["kind"] = "position"
    data["y"].attrs["kind"] = "position"
    data["Observation"].attrs["kind"] = "observation"

    return fmskill.comparison.Comparer(matched_data=data, raw_mod_data=raw_data)


def test_comparer_properties(pc):
    assert pc.n_models == 2
    assert pc.n_points == 5


def test_comparer_where(pc):
    pc2 = pc.where(pc.data.Observation > 2.5)
    assert pc2.n_points == 3
    assert pc2.data.Observation.values.tolist() == [3.0, 4.0, 5.0]

def test_comparer_where_empty(pc):
    pc2 = pc.where(pc.data.Observation > 10.0)
    assert pc2.n_points == 0

def test_comparer_where_derived(pc):
    pc.data["derived"] = pc.data.m1 + pc.data.m2
    pc2 = pc.where(pc.data.derived > 5.0)
    assert pc2.n_points == 3
    assert pc2.data.Observation.values.tolist() == [3.0, 4.0, 5.0]
