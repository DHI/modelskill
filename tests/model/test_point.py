from datetime import datetime
import numpy as np
import pytest

import mikeio
from modelskill import ModelResult, Quantity
from modelskill.model import PointModelResult


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


def test_point_dfs0(fn_point_eq):
    fn = fn_point_eq

    ds = mikeio.read(fn)
    assert len(ds.items) == 1
    assert ds[0].shape[0] == 2017
    assert ds[0].start_time == datetime(2019, 4, 8, 0, 0, 0)
    assert ds[0].end_time == datetime(2019, 4, 15, 0, 0, 0)
    assert ds[0].dropna().shape[0] == 1875

    mr1 = PointModelResult(fn, item=0)
    assert mr1.name == "smhi_2095_klagshamn_shifted"

    assert isinstance(mr1, PointModelResult)
    assert mr1.start_time == datetime(2019, 4, 8, 0, 10, 0)  # first non-NaN
    assert mr1.end_time == datetime(2019, 4, 14, 23, 35, 0)  # last non-NaN
    assert len(mr1.data) == 1875

    mr2 = PointModelResult(fn, item="Viken: Surface elevation")
    assert mr1.data.equals(mr2.data)

    mr3 = PointModelResult(fn)  # default to item=0
    assert mr1.data.equals(mr3.data)


def test_point_dfs0_noneq(fn_point_noneq):
    fn = fn_point_noneq

    ds = mikeio.read(fn)
    assert len(ds.items) == 1
    assert ds.items[0].name == "Water Level"
    assert len(ds[0].time) == 200
    mr1 = PointModelResult(fn, item=0, name="test")
    assert isinstance(mr1, PointModelResult)
    assert mr1.name == "test"
    assert mr1.start_time == datetime(2015, 1, 1, 1, 0, 0)
    assert mr1.end_time == datetime(2015, 1, 9, 8, 0, 0)
    assert len(mr1.data) == 198  # 200 - 2 NaNs


def test_point_dfs0_multi_item(fn_point_eq2):
    fn = fn_point_eq2

    ds = mikeio.read(fn)
    assert len(ds.items) == 12
    assert len(ds[0].time) == 2017
    mr1 = PointModelResult(fn, item=2, name="test")
    assert isinstance(mr1, PointModelResult)
    assert mr1.name == "test"
    assert mr1.start_time == datetime(2018, 3, 4, 0, 0, 0)
    assert mr1.end_time == datetime(2018, 3, 11, 0, 0, 0)
    assert len(mr1.data) == 2017

    mr2 = PointModelResult(fn, item="Drogden: Surface elevation")
    assert isinstance(mr2, PointModelResult)
    assert mr2.name == "TS"  # default to filename
    assert np.all(mr2.data.values == mr1.data.values)

    with pytest.raises(ValueError):
        PointModelResult(fn, name="test")


def test_point_dfs0_last_item(fn_point_eq2):
    fn = fn_point_eq2
    mr1 = PointModelResult(fn, item=-1, name="test")
    assert isinstance(mr1, PointModelResult)
    assert mr1.name == "test"


def test_point_df_item(point_df):
    df = point_df
    df["ones"] = 1.0

    mr1 = PointModelResult(df, item=0)
    assert isinstance(mr1, PointModelResult)
    assert mr1.start_time == datetime(2015, 1, 1, 1, 0, 0)
    assert mr1.end_time == datetime(2015, 1, 9, 8, 0, 0)
    assert mr1.name == "Water Level"

    # item as string
    mr2 = PointModelResult(df, item="Water Level")
    assert len(mr2.data) == len(mr1.data)

    mr3 = PointModelResult(df[["Water Level"]])
    assert len(mr3.data) == len(mr1.data)

    # Series
    mr4 = PointModelResult(df["Water Level"])
    assert len(mr4.data) == len(mr1.data)
    assert np.all(mr4.data.values == mr1.data.values)


def test_point_df_itemInfo(point_df):
    df = point_df
    df["ones"] = 1.0
    # itemInfo = mikeio.EUMType.Surface_Elevation
    mr1 = ModelResult(
        df,
        item="Water Level",
        quantity=Quantity(name="Surface elevation", unit="meter"),
    )
    assert mr1.quantity.name == "Surface elevation"


def test_point_df(point_df):
    df = point_df
    df["ones"] = 1.0

    mr1 = PointModelResult(df, item=0)
    assert isinstance(mr1, PointModelResult)
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
