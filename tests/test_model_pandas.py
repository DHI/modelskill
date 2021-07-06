from datetime import datetime
import pandas as pd
import pytest

from mikeio import Dfs0, eum
from fmskill import ModelResult, PointObservation, TrackObservation
from fmskill.model.abstract import ModelResultInterface
from fmskill.model import DataFrameModelResult, DataFrameModelResultItem
from fmskill.model.pandas import (
    DataFrameTrackModelResult,
    DataFrameTrackModelResultItem,
)


@pytest.fixture
def klagshamn():
    fn = "tests/testdata/smhi_2095_klagshamn.dfs0"
    df = Dfs0(fn).to_dataframe()
    return df


@pytest.fixture
def track_df():
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    return pd.read_csv(fn, index_col=0, parse_dates=True)


def test_df_modelresultitem(klagshamn):
    df = klagshamn
    df["ones"] = 1.0

    mr1 = DataFrameModelResultItem(df, item=0)
    assert isinstance(mr1, ModelResultInterface)
    assert mr1.start_time == datetime(2015, 1, 1, 1, 0, 0)
    assert mr1.end_time == datetime(2020, 9, 28, 0, 0, 0)
    assert mr1.name == "Water Level"

    # item as string
    mr2 = DataFrameModelResultItem(df, item="Water Level")
    assert len(mr2.df) == len(mr1.df)

    mr3 = DataFrameModelResultItem(df[["Water Level"]])
    assert len(mr3.df) == len(mr1.df)

    # Series
    mr4 = DataFrameModelResultItem(df["Water Level"])
    assert len(mr4.df) == len(mr1.df)


def test_df_modelresult(klagshamn):
    df = klagshamn
    df["ones"] = 1.0

    mr1 = DataFrameModelResult(df)
    assert not isinstance(mr1, ModelResultInterface)
    assert mr1.start_time == datetime(2015, 1, 1, 1, 0, 0)
    assert mr1.end_time == datetime(2020, 9, 28, 0, 0, 0)
    assert mr1.name == "model"

    mr2 = mr1["Water Level"]
    assert len(mr2.df) == len(mr1.df)


def test_track_df_modelresultitem(track_df):
    df = track_df
    mr1 = DataFrameTrackModelResultItem(df, item=2)
    assert isinstance(mr1, ModelResultInterface)
    assert "Item: surface_elevation" in repr(mr1)

    # item as string
    mr2 = DataFrameModelResultItem(df, item="surface_elevation")
    assert len(mr2.df) == len(mr1.df)

    mr3 = DataFrameModelResultItem(df[["surface_elevation"]])
    assert len(mr3.df) == len(mr1.df)


def test_track_df_modelresult(track_df):
    df = track_df
    mr1 = DataFrameTrackModelResult(df)
    assert not isinstance(mr1, ModelResultInterface)
    assert len(df.columns) == 5
    assert len(mr1.item_names) == 3

    mr2 = mr1["surface_elevation"]
    assert len(mr2.df) == len(mr1.df)
    assert isinstance(mr2, ModelResultInterface)


def test_track_df_model_extract(track_df):
    df = track_df
    mr1 = DataFrameTrackModelResultItem(df, item=2)
    o1 = TrackObservation(df, item=2)
    c = mr1.extract_observation(o1)
    assert c is not None
