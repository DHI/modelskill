import pandas as pd
import pytest
import mikeio

import modelskill as ms


@pytest.fixture
def track_df():
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    return pd.read_csv(fn, index_col=0, parse_dates=True)


@pytest.fixture
def df_aux():
    df = pd.DataFrame(
        {
            "WL": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x": [10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
            "y": [55.1, 55.2, 55.3, 55.4, 55.5, 55.6],
            "aux1": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            "aux2": [1.2, 2.2, 3.2, 4.2, 5.2, 6.2],
            "time": pd.date_range("2019-01-01", periods=6, freq="D"),
        }
    ).set_index("time")
    return df


@pytest.fixture
def track_from_dfs0():
    fn = "tests/testdata/NorthSeaHD_extracted_track.dfs0"
    return mikeio.open(fn).to_dataframe()


def test_track_df(track_df):
    df = track_df
    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        mr1 = ms.TrackModelResult(df, item=2)

    # item as string
    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        mr2 = ms.TrackModelResult(df, item="surface_elevation")
    assert len(mr2.data) == len(mr1.data)


def test_track_df_iteminfo(track_df):
    df = track_df
    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        mr1 = ms.TrackModelResult(
            df, item=2, quantity=ms.Quantity(name="Surface Elevation", unit="meter")
        )

    assert mr1.quantity.name == "Surface Elevation"


def test_track_df_modelresult(track_df):
    df = track_df
    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        mr1 = ms.TrackModelResult(df, item=2)
    assert len(mr1.data) == 1

    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        mr3 = ms.TrackModelResult(df, item=2)
    assert len(mr3.data) == len(mr1.data)


def test_track_from_dfs0_df_modelresult(track_from_dfs0):
    df = track_from_dfs0
    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        mr1 = ms.TrackModelResult(df, item=-1)
    assert len(mr1.data) == 1


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
    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        mr0 = ms.model_result(df, gtype="track")
    assert "x" in mr0.data.coords

    mr = ms.model_result(
        df, gtype="track", x_item=1, y_item="lat", item="surface_elevation"
    )
    assert "x" in mr.data.coords

    # Rename
    df = df.copy()
    df.columns = ["wl", "longitude", "latitude"]
    df["ones"] = 1.0  # add extra column

    with pytest.raises(ValueError):
        # cannot default anymore - more than 3 columns
        ms.model_result(df, gtype="track")

    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        mr3 = ms.model_result(
            df, item="wl", gtype="track", x_item="longitude", y_item="latitude"
        )
    assert "x" in mr3.data.coords
    assert "y" in mr3.data.coords


def test_track_aux_items(df_aux):
    o = ms.TrackModelResult(
        df_aux, item="WL", x_item="x", y_item="y", aux_items=["aux1"]
    )
    assert "aux1" in o.data
    assert o.data["aux1"].values[0] == 1.1

    o = ms.TrackModelResult(df_aux, item="WL", x_item="x", y_item="y", aux_items="aux1")
    assert "aux1" in o.data
    assert o.data["aux1"].values[0] == 1.1


def test_track_aux_items_multiple(df_aux):
    o = ms.TrackModelResult(
        df_aux, item="WL", x_item="x", y_item="y", aux_items=["aux2", "aux1"]
    )
    assert "aux1" in o.data
    assert o.data["aux1"].values[0] == 1.1
    assert "aux2" in o.data
    assert o.data["aux2"].values[0] == 1.2


def test_track_aux_items_fail(df_aux):
    with pytest.raises(KeyError):
        ms.TrackModelResult(
            df_aux, item="WL", x_item="x", y_item="y", aux_items=["aux1", "aux3"]
        )

    with pytest.raises(ValueError):
        ms.TrackModelResult(df_aux, item="WL", x_item="x", y_item="y", aux_items=["x"])
