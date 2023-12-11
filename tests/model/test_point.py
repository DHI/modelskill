from datetime import datetime
import numpy as np
import pandas as pd
import pytest

import mikeio
import modelskill as ms


@pytest.fixture
def fn_point_noneq():
    # 1 item with 200 timesteps 2 missing values
    return "tests/testdata/smhi_2095_klagshamn_200.dfs0"


@pytest.fixture
def fn_point_eq():
    # 1 item with 2017 timesteps many gaps
    return "tests/testdata/smhi_2095_klagshamn_shifted.dfs0"


@pytest.fixture
def fn_point_eq2():
    # 12 items with 2017 timesteps
    return "tests/testdata/TS.dfs0"


@pytest.fixture
def point_df(fn_point_noneq):
    return mikeio.open(fn_point_noneq).to_dataframe()


@pytest.fixture
def df_aux():
    df = pd.DataFrame(
        {
            "WL": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "aux1": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            "aux2": [1.2, 2.2, 3.2, 4.2, 5.2, 6.2],
            "time": pd.date_range("2019-01-01", periods=6, freq="D"),
        }
    ).set_index("time")
    return df


def test_point_dfs0(fn_point_eq):
    fn = fn_point_eq

    ds = mikeio.read(fn)
    assert len(ds.items) == 1
    assert ds[0].shape[0] == 2017
    assert ds[0].start_time == datetime(2019, 4, 8, 0, 0, 0)
    assert ds[0].end_time == datetime(2019, 4, 15, 0, 0, 0)
    assert ds[0].dropna().shape[0] == 1875

    mr1 = ms.PointModelResult(fn, item=0)
    assert mr1.name == "smhi_2095_klagshamn_shifted"

    assert isinstance(mr1, ms.PointModelResult)
    assert mr1.start_time == datetime(2019, 4, 8, 0, 10, 0)  # first non-NaN
    assert mr1.end_time == datetime(2019, 4, 14, 23, 35, 0)  # last non-NaN
    assert mr1.n_points == 1875

    mr2 = ms.PointModelResult(fn, item="Viken: Surface elevation")
    assert mr1.data.equals(mr2.data)

    mr3 = ms.PointModelResult(fn)  # default to item=0
    assert mr1.data.equals(mr3.data)


def test_point_dfs0_noneq(fn_point_noneq):
    fn = fn_point_noneq

    ds = mikeio.read(fn)
    assert len(ds.items) == 1
    assert ds.items[0].name == "Water Level"
    assert len(ds[0].time) == 200
    mr1 = ms.PointModelResult(fn, item=0, name="test")
    assert isinstance(mr1, ms.PointModelResult)
    assert mr1.name == "test"
    assert mr1.start_time == datetime(2015, 1, 1, 1, 0, 0)
    assert mr1.end_time == datetime(2015, 1, 9, 8, 0, 0)
    assert mr1.n_points == 198  # 200 - 2 NaNs


def test_point_dfs0_multi_item(fn_point_eq2):
    fn = fn_point_eq2

    ds = mikeio.read(fn)
    assert len(ds.items) == 12
    assert len(ds[0].time) == 2017
    mr1 = ms.PointModelResult(fn, item=2, name="test")
    assert isinstance(mr1, ms.PointModelResult)
    assert mr1.name == "test"
    assert mr1.start_time == datetime(2018, 3, 4, 0, 0, 0)
    assert mr1.end_time == datetime(2018, 3, 11, 0, 0, 0)
    assert mr1.n_points == 2017

    item_name = "Drogden: Surface elevation"
    mr2 = ms.PointModelResult(fn, item=item_name)
    assert isinstance(mr2, ms.PointModelResult)
    assert mr2.name == "TS"  # default to filename
    assert np.all(mr2.data[mr2.name].values == mr1.data[mr1.name].values)

    with pytest.raises(ValueError):
        ms.PointModelResult(fn, name="test")


def test_point_dfs0_last_item(fn_point_eq2):
    fn = fn_point_eq2
    mr1 = ms.PointModelResult(fn, item=-1, name="test")
    assert isinstance(mr1, ms.PointModelResult)
    assert mr1.name == "test"


def test_point_df_item(point_df):
    df = point_df
    df["ones"] = 1.0

    mr1 = ms.PointModelResult(df, item=0)
    assert isinstance(mr1, ms.PointModelResult)
    assert mr1.start_time == datetime(2015, 1, 1, 1, 0, 0)
    assert mr1.end_time == datetime(2015, 1, 9, 8, 0, 0)
    assert mr1.name == "Water Level"

    # item as string
    mr2 = ms.PointModelResult(df, item="Water Level")
    assert mr2.n_points == mr1.n_points

    mr3 = ms.PointModelResult(df[["Water Level"]])
    assert mr3.n_points == mr1.n_points

    # Series
    mr4 = ms.PointModelResult(df["Water Level"])
    assert mr4.n_points == mr1.n_points
    assert np.all(mr4.data[mr4.name].values == mr1.data[mr1.name].values)


def test_point_df_itemInfo(point_df):
    df = point_df
    df["ones"] = 1.0
    # itemInfo = mikeio.EUMType.Surface_Elevation
    mr1 = ms.model_result(
        df,
        item="Water Level",
        quantity=ms.Quantity(name="Surface elevation", unit="meter"),
    )
    assert mr1.quantity.name == "Surface elevation"


def test_point_df(point_df):
    df = point_df
    df["ones"] = 1.0

    mr1 = ms.PointModelResult(df, item=0)
    assert isinstance(mr1, ms.PointModelResult)
    assert mr1.start_time == datetime(2015, 1, 1, 1, 0, 0)
    assert mr1.end_time == datetime(2015, 1, 9, 8, 0, 0)
    assert mr1.name == "Water Level"


# TODO
# def test_point_df_compare(point_df):
#     df = point_df
#     mr1 = DataFramePointModelResultItem(df, item=0)
#     o1 = PointObservation(df, item=0)
#     c = mr1.compare(o1)
#     assert c.score() == 0.0  # o1=mr1
#     assert c.n_points == len(o1.data.dropna())


def test_point_model_data_can_be_persisted_as_netcdf(point_df, tmp_path):
    mr = ms.PointModelResult(point_df, item=0)

    mr.data.to_netcdf(tmp_path / "test.nc")


def test_point_aux_items(df_aux):
    o = ms.PointModelResult(df_aux, item="WL", aux_items=["aux1"])
    assert "aux1" in o.data
    assert o.data["aux1"].values[0] == 1.1

    o = ms.PointModelResult(df_aux, item="WL", aux_items="aux1")
    assert "aux1" in o.data
    assert o.data["aux1"].values[0] == 1.1


def test_point_aux_items_fail(df_aux):
    with pytest.raises(KeyError):
        ms.PointModelResult(df_aux, item="WL", aux_items=["aux1", "aux3"])

    with pytest.raises(ValueError):
        ms.PointModelResult(df_aux, item="WL", aux_items="WL")


def test_point_aux_items_multiple(df_aux):
    o = ms.PointModelResult(df_aux, item="WL", aux_items=["aux1", "aux2"])
    assert "aux1" in o.data
    assert "aux2" in o.data
    assert o.data["aux1"].values[0] == 1.1
    assert o.data["aux2"].values[0] == 1.2


def test_point_modelresult_must_have_unique_and_monotonically_increasing_time():
    df_dup = pd.DataFrame(
        {"wl": [0.0, 1.0, 2.0]},
        index=pd.DatetimeIndex(
            ["2015-01-01", "2015-01-01", "2015-01-02"]  # same as previous
        ),
    )

    with pytest.raises(ValueError):
        ms.PointModelResult(df_dup, item=0)

    df_up_and_down = pd.DataFrame(
        {"wl": [0.0, 1.0, 2.0]},
        index=pd.DatetimeIndex(["2015-01-01", "2015-01-02", "2015-01-01"]),
    )

    with pytest.raises(ValueError):
        ms.PointModelResult(df_up_and_down, item=0)
