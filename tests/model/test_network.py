import pytest
import mikeio1d

import numpy as np
import pandas as pd
import modelskill as ms

parse_network = ms.timeseries._parse_network_input


@pytest.fixture
def res1d_datapath() -> str:
    return "tests/testdata/network.res1d"


@pytest.fixture
def res1d_object(res1d_datapath) -> mikeio1d.Res1D:
    return mikeio1d.open(res1d_datapath)


def test_read_quantity_by_node(res1d_object):
    series = parse_network(res1d_object, variable="Water Level", node=3)
    df = res1d_object.read()
    assert isinstance(series, pd.Series)
    assert series.name == "WaterLevel"
    np.testing.assert_allclose(df["WaterLevel:3"].values, series.values)


@pytest.mark.parametrize(
    "network_kwargs",
    [
        dict(gridpoint="end"),
        dict(gridpoint=2),
        dict(chainage=47.683),
        dict(chainage="47.683"),
    ],
)
def test_read_quantity_by_reach(res1d_object, network_kwargs):
    series = parse_network(
        res1d_object, variable="Water Level", reach="100l1", **network_kwargs
    )
    df = res1d_object.read()
    assert isinstance(series, pd.Series)
    assert series.name == "WaterLevel"
    np.testing.assert_allclose(df["WaterLevel:100l1:47.6827"].values, series.values)


def test_node_and_reach_as_arguments(res1d_object):
    with pytest.raises(
        ValueError, match="Item can only be specified either by node or by reach"
    ):
        parse_network(res1d_object, variable="Water Level", reach="100l1", node=2)
