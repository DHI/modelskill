import pytest
import pandas as pd
import numpy as np

import modelskill as ms


@pytest.fixture
def c2():
    return "tests/testdata/SW/Alti_c2_Dutch.dfs0"


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
def obs_tiny_df4():
    time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:04",
        ]
    )
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([11.0, 12.0, 13.0, 14.0])
    val = np.array([1.0, 2.0, 4.0, 6.0])
    return pd.DataFrame(data={"x": x, "y": y, "alti": val}, index=time)


@pytest.fixture
def obs_tiny_df_duplicates():
    time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:02",  # duplicate time (not spatially)
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:03",  # duplicate time (not spatially)
            "2017-10-27 13:00:04",
        ]
    )
    x = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0])
    y = np.array([11.0, 12.0, 12.5, 13.0, 13.5, 14.0])
    val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return pd.DataFrame(data={"x": x, "y": y, "alti": val}, index=time)


def test_tiny_obs_offset(obs_tiny_df_duplicates):
    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        obs_tiny = ms.TrackObservation(
            obs_tiny_df_duplicates,
            item="alti",
            x_item="x",
            y_item="y",
            keep_duplicates="offset",
        )
    assert len(obs_tiny) == 6
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02.001",
            "2017-10-27 13:00:02.002",
            "2017-10-27 13:00:03.003",
            "2017-10-27 13:00:03.004",
            "2017-10-27 13:00:04",
        ]
    )
    expected_x = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0])
    expected_y = np.array([11.0, 12.0, 12.5, 13.0, 13.5, 14.0])
    expected_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert obs_tiny.time.equals(expected_time)
    assert np.all(obs_tiny.x == expected_x)
    assert np.all(obs_tiny.y == expected_y)
    assert np.all(obs_tiny.values == expected_val)


def test_tiny_obs_first(obs_tiny_df_duplicates):
    with pytest.warns(UserWarning, match="Removed 2 duplicate timestamps"):
        obs_tiny = ms.TrackObservation(
            obs_tiny_df_duplicates,
            item="alti",
            x_item="x",
            y_item="y",
            keep_duplicates="first",
        )

    assert len(obs_tiny) == 4
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:04",
        ]
    )
    expected_x = np.array([1.0, 2.0, 3.0, 4.0])
    expected_y = np.array([11.0, 12.0, 13.0, 14.0])
    expected_val = np.array([1.0, 2.0, 4.0, 6.0])
    assert obs_tiny.time.equals(expected_time)
    assert np.all(obs_tiny.x == expected_x)
    assert np.all(obs_tiny.y == expected_y)
    assert np.all(obs_tiny.values == expected_val)


def test_tiny_obs_last(obs_tiny_df_duplicates):
    with pytest.warns(UserWarning, match="Removed 2 duplicate timestamps"):
        obs_tiny = ms.TrackObservation(
            obs_tiny_df_duplicates,
            item="alti",
            x_item="x",
            y_item="y",
            keep_duplicates="last",
        )

    assert len(obs_tiny) == 4
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:04",
        ]
    )
    expected_x = np.array([1.0, 2.5, 3.5, 4.0])
    expected_y = np.array([11.0, 12.5, 13.5, 14.0])
    expected_val = np.array([1.0, 3.0, 5.0, 6.0])
    assert obs_tiny.time.equals(expected_time)
    assert np.all(obs_tiny.x == expected_x)
    assert np.all(obs_tiny.y == expected_y)
    assert np.all(obs_tiny.values == expected_val)


def test_tiny_obs_False(obs_tiny_df_duplicates):
    with pytest.warns(UserWarning, match="Removed 4 duplicate timestamps"):
        obs_tiny = ms.TrackObservation(
            obs_tiny_df_duplicates,
            item="alti",
            x_item="x",
            y_item="y",
            keep_duplicates=False,
        )

    assert len(obs_tiny) == 2
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:04",
        ]
    )
    expected_x = np.array([1.0, 4.0])
    expected_y = np.array([11.0, 14.0])
    expected_val = np.array([1.0, 6.0])
    assert obs_tiny.time.equals(expected_time)
    assert np.all(obs_tiny.x == expected_x)
    assert np.all(obs_tiny.y == expected_y)
    assert np.all(obs_tiny.values == expected_val)


def test_read(c2):
    o1 = ms.TrackObservation(c2, item=2, name="c2")
    assert o1.n_points == 299  # 298 + 1 NaN
    assert len(o1.x) == o1.n_points
    assert o1.name == "c2"
    assert np.nanmax(o1.values) == pytest.approx(17.67)
    # o2 = TrackObservation(c2, item=2, name="c2", units="inches/hour")
    ms.TrackObservation(
        c2,
        item=2,
        name="c2",
        quantity=ms.Quantity(name="Wind speed", unit="inches/hour"),
    )
    assert "x" in o1.data
    assert "y" in o1.data


def test_from_df():
    n = 5

    df = pd.DataFrame(
        {
            "t": pd.date_range("2010-01-01", freq="10s", periods=n),
            "x": np.linspace(0, 10, n),
            "y": np.linspace(45000, 45100, n),
            "swh": [0.1, 0.3, 0.4, 0.5, 0.3],
        }
    )

    df = df.set_index("t")

    t1 = ms.TrackObservation(df, item="swh", name="fake")
    assert t1.n_points == n


def test_observation_factory(obs_tiny_df4):
    o = ms.observation(obs_tiny_df4, x_item="x", y_item="y", item="alti")
    assert isinstance(o, ms.TrackObservation)

    with pytest.warns(UserWarning, match="Could not guess geometry"):
        o = ms.observation(obs_tiny_df4, item="alti", name="Klagshamn")
        assert not isinstance(o, ms.TrackObservation)  # ! defaults to PointObservation


def test_non_unique_index():
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    df = pd.read_csv(fn, index_col=0, parse_dates=True)
    assert not df.index.is_unique
    assert df.index[160] == df.index[161]

    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        o = ms.TrackObservation(df, item=2, keep_duplicates="offset")
    o_index = o.data.time.to_index()
    assert o_index.is_unique
    assert not df.index.is_unique  # did not change input data
    assert o_index[160].to_pydatetime().microsecond == 1000
    assert o_index[161].to_pydatetime().microsecond == 2000


def test_trackobservation_item_dfs0(c2):
    with pytest.raises(ValueError, match="more than 3 items, but item was not given"):
        ms.TrackObservation(c2)

    o1 = ms.TrackObservation(c2, item=2)
    assert o1.n_points == 299  # 298 + 1 NaN

    o2 = ms.TrackObservation(c2, item="swh")
    assert o2.n_points == 299  # 298 + 1 NaN


def test_trackobservation_item_csv():
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    df = pd.read_csv(fn, index_col=0, parse_dates=True)

    with pytest.raises(ValueError, match="Input has only 2 items"):
        ms.TrackObservation(df[["lon", "surface_elevation"]])

    with pytest.raises(ValueError, match="more than 3 items, but item was not given"):
        ms.TrackObservation(df)

    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        o1 = ms.TrackObservation(df, item=-1)
    assert o1.n_points == 1093  # 1115
    assert list(o1.data.data_vars)[-1] == "wind_speed"

    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        o2 = ms.TrackObservation(df, item="significant_wave_height")
    assert o2.n_points == 1093  # 1115  # including 1 NaN


def test_hist(c2):
    o1 = ms.TrackObservation(c2, item=2)
    o1.plot.hist()

    o1.plot.hist(bins=20, title="new_title", color="red")


def test_trackobservation_x_y_item(c2):
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    df_in = pd.read_csv(fn, index_col=0, parse_dates=True)
    cols = ["lat", "surface_elevation", "lon", "wind_speed"]
    df = df_in[cols]  # re-order columns

    with pytest.raises(ValueError, match="more than 3 items, but item was not given"):
        ms.TrackObservation(df)

    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        o1 = ms.TrackObservation(df, item=-1, x_item="lon", y_item="lat")
    assert o1.n_points == 1093  # 1115
    assert list(o1.data.data_vars)[-1] == "wind_speed"

    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        o2 = ms.TrackObservation(df, item="surface_elevation", x_item=2, y_item=0)
    assert o2.n_points == 1093  # 1115  # including 5 NaN

    with pytest.raises(ValueError, match="Duplicate items"):
        ms.TrackObservation(df, item=-1, x_item="lon", y_item="lon")

    cols = ["lat", "surface_elevation", "lon"]
    df = df_in[cols]
    with pytest.raises(ValueError, match="Duplicate items"):
        ms.TrackObservation(df, x_item="lon", y_item="lat")


def test_force_keyword_args(c2):
    with pytest.raises(TypeError):
        ms.TrackObservation(c2, 2, "c2")


def test_track_data_can_be_persisted_as_netcdf(c2, tmp_path):
    t = ms.TrackObservation(c2, item=2, name="c2")

    t.data.to_netcdf(tmp_path / "test.nc")


def test_track_attrs(obs_tiny_df4):
    o1 = ms.TrackObservation(obs_tiny_df4, item="alti", attrs={"a1": "v1"})
    assert o1.data.attrs["a1"] == "v1"


def test_track_attrs_not_allowed(obs_tiny_df4):
    with pytest.raises(ValueError, match="attrs key gtype not allowed"):
        ms.PointObservation(obs_tiny_df4, item="alti", attrs={"gtype": "v1"})


def test_track_aux_items(df_aux):
    o = ms.TrackObservation(
        df_aux, item="WL", x_item="x", y_item="y", aux_items=["aux1"]
    )
    assert "aux1" in o.data
    assert o.data["aux1"].values[0] == 1.1

    o = ms.TrackObservation(df_aux, item="WL", x_item="x", y_item="y", aux_items="aux1")
    assert "aux1" in o.data
    assert o.data["aux1"].values[0] == 1.1


def test_track_aux_items_multiple(df_aux):
    o = ms.TrackObservation(
        df_aux, item="WL", x_item="x", y_item="y", aux_items=["aux2", "aux1"]
    )
    assert "aux1" in o.data
    assert o.data["aux1"].values[0] == 1.1
    assert "aux2" in o.data
    assert o.data["aux2"].values[0] == 1.2


def test_track_aux_items_fail(df_aux):
    with pytest.raises(KeyError):
        ms.TrackObservation(
            df_aux, item="WL", x_item="x", y_item="y", aux_items=["aux1", "aux3"]
        )

    with pytest.raises(ValueError):
        ms.TrackObservation(df_aux, item="WL", x_item="x", y_item="y", aux_items=["x"])


def test_track_basic_repr(df_aux):
    # Some basic test to see that repr does not fail
    o = ms.TrackObservation(
        df_aux, item="WL", x_item="x", y_item="y", aux_items=["aux1"]
    )
    assert "TrackObservation" in repr(o)
    assert "WL" in repr(o)
    assert "aux1" in repr(o)
