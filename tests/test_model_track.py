import numpy as np
import pandas as pd
import pytest
import mikeio

from fmskill import ModelResult, TrackObservation
from fmskill.model import TrackModelResult, protocols


@pytest.fixture
def track_df():
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    return pd.read_csv(fn, index_col=0, parse_dates=True)


@pytest.fixture
def track_from_dfs0():
    fn = "tests/testdata/NorthSeaHD_extracted_track.dfs0"
    return mikeio.open(fn).to_dataframe()


def test_track_df(track_df):
    df = track_df
    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        mr1 = TrackModelResult(df, item=2)
    assert isinstance(mr1, protocols.ModelResult)
    assert "Item: surface_elevation" in repr(mr1)

    # item as string
    mr2 = TrackModelResult(df, item="surface_elevation")
    assert len(mr2.data) == len(mr1.data)
    assert mr2.itemInfo == mikeio.ItemInfo(mikeio.EUMType.Undefined)

    # mr3 = DataFramePointModelResultItem(df[["surface_elevation"]])
    # assert len(mr3.data) == len(mr1.data)
    # assert mr3.itemInfo == mikeio.ItemInfo(mikeio.EUMType.Undefined)


def test_track_df_iteminfo(track_df):
    df = track_df
    itemInfo = mikeio.EUMType.Surface_Elevation
    mr1 = TrackModelResult(df, item="surface_elevation", itemInfo=itemInfo)
    assert mr1.itemInfo == mikeio.ItemInfo(mikeio.EUMType.Surface_Elevation)

    itemInfo = mikeio.ItemInfo("WL", mikeio.EUMType.Surface_Elevation)
    mr2 = TrackModelResult(df, item="surface_elevation", itemInfo=itemInfo)
    assert mr2.itemInfo == mikeio.ItemInfo("WL", mikeio.EUMType.Surface_Elevation)


def test_track_df_modelresult(track_df):
    df = track_df
    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        mr1 = TrackModelResult(df, item=2)
    assert isinstance(mr1, protocols.ModelResult)
    assert len(mr1.data.columns) == 3
    assert mr1.item == "surface_elevation"

    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        mr3 = TrackModelResult(df, item=2)
    assert len(mr3.data) == len(mr1.data)
    assert isinstance(mr3, protocols.ModelResult)


def test_track_from_dfs0_df_modelresult(track_from_dfs0):
    df = track_from_dfs0
    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        mr1 = TrackModelResult(df, item=-1)
    assert isinstance(mr1, protocols.ModelResult)
    assert len(mr1.data.columns) == 3


def test_track_df_default_items(track_df):
    df = track_df
    # Reorder columns
    df = df[
        [
            "surface_elevation",
            "lon",
            "lat",
        ]
    ]

    # Which columns are used for position, lon and lat?
    # will default to 0,1,2 (bad idea in this case)
    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        mr0 = ModelResult(df, geometry_type="track")
    assert mr0.data.columns[0] == "x"  # has been renamed
    # assert np.all(mr0.data.values == df.dropna().values)
    # assert np.all(mr0.data["x"].values == df["surface_elevation"].values)
    # assert np.all(mr0.data["lat"].values == df["lat"].values)
    assert mr0.item == "lat"

    with pytest.raises(ValueError):
        # cannot default item as x_item and y_item are not default
        mr1 = ModelResult(df, geometry_type="track", x_item=1, y_item="lat")
    # assert isinstance(mr1, protocols.ModelResult)
    # assert mr1.item_name == "surface_elevation"

    # Rename
    df = df.copy()
    df.columns = ["wl", "longitude", "latitude"]
    df["ones"] = 1.0  # add extra column

    with pytest.raises(ValueError):
        # cannot default anymore - more than 3 columns
        ModelResult(df, geometry_type="track")

    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        mr3 = ModelResult(
            df, item="wl", geometry_type="track", x_item="longitude", y_item="latitude"
        )
    assert isinstance(mr3, protocols.ModelResult)
    # assert np.all(mr3.data.values == mr1.data.values)


# TODO:
# def test_track_df_compare(track_df):
#     df = track_df
#     with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
#         mr1 = ModelResult(df, item=2)
#     with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
#         o1 = TrackObservation(df, item=2)
#     c = mr1.compare(o1)
#     assert c.score() == 0.0  # o1=mr1
#     assert len(o1.data.dropna()) == 1110
#     assert c.n_points == 1110
