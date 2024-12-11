import pytest
import pandas as pd
import numpy as np

import modelskill as ms


@pytest.fixture
def obs_tiny_df():
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


@pytest.fixture
def obs_tiny(obs_tiny_df):
    with pytest.warns(UserWarning, match="Removed 2 duplicate timestamps"):
        o = ms.TrackObservation(obs_tiny_df, item="alti", x_item="x", y_item="y")
    return o


@pytest.fixture
def mod_tiny3():
    time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:02",  # duplicate
            "2017-10-27 13:00:03",
        ]
    )
    x = np.array([2.0, 2.5, 3.0])
    y = np.array([12.0, 12.5, 13.0])
    val = np.array([2.1, 3.1, 4.1])
    df = pd.DataFrame(data={"x": x, "y": y, "m1": val}, index=time)
    with pytest.warns(UserWarning, match="Removed 1 duplicate timestamps"):
        mr = ms.TrackModelResult(df, item="m1", x_item="x", y_item="y")
    return mr


@pytest.fixture
def mod_tiny_3last():
    time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:03",  # duplicate time (not spatially)
            "2017-10-27 13:00:04",
        ]
    )
    x = np.array([3.0, 3.5, 4.0])
    y = np.array([13.0, 13.5, 14.0])
    val = np.array([4.1, 5.1, 6.1])
    df = pd.DataFrame(data={"x": x, "y": y, "m1": val}, index=time)
    with pytest.warns(UserWarning, match="Removed 1 duplicate timestamps"):
        mr = ms.TrackModelResult(df, item="m1", x_item="x", y_item="y")
    return mr


@pytest.fixture
def mod_tiny_unique():
    """Model match observation, except for duplicate time (removed)"""
    time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02",
            # "2017-10-27 13:00:02",  # duplicate time (not spatially)
            "2017-10-27 13:00:03",
            # "2017-10-27 13:00:03",  # duplicate time (not spatially)
            "2017-10-27 13:00:04",
        ]
    )
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([11.0, 12.0, 13.0, 14.0])
    val = np.array([1.1, 2.1, 4.1, 6.1])
    df = pd.DataFrame(data={"x": x, "y": y, "m1": val}, index=time)
    return ms.TrackModelResult(df, item="m1", x_item="x", y_item="y")


@pytest.fixture
def mod_tiny_rounding_error():
    """Model match observation, but with rounding error on x,y"""
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
    x = np.array([1.0, 2.0, 2.50001, 3.0, 3.50001, 4.0])
    y = np.array([11.0, 12.0, 12.5, 13.0, 13.50001, 14.0])
    val = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1])
    df = pd.DataFrame(data={"x": x, "y": y, "m1": val}, index=time)
    with pytest.warns(UserWarning, match="duplicate"):
        mr = ms.TrackModelResult(df, item="m1", x_item="x", y_item="y")
    return mr


@pytest.fixture
def mod_tiny_longer():
    """Model match observation, but with more data"""
    time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:04",
            "2017-10-27 13:00:04",
            "2017-10-27 13:00:05",
        ]
    )
    x = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    y = np.array([11.0, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0])
    val = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1])
    df = pd.DataFrame(data={"x": x, "y": y, "m1": val}, index=time)
    # with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
    mr = ms.TrackModelResult(df, item="m1", x_item="x", y_item="y")
    return mr


# TODO: add some with missing values


def test_tiny_mod3(obs_tiny, mod_tiny3):
    cmp = ms.match(obs_tiny, mod_tiny3)
    assert cmp.n_points == 2
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:03",
        ]
    )
    assert cmp.time.equals(expected_time)
    assert np.all(cmp.x == np.array([2.0, 3.0]))


def test_tiny_mod_3last(obs_tiny, mod_tiny_3last):
    cmp = ms.match(obs_tiny, mod_tiny_3last)
    assert cmp.n_points == 2
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:04",
        ]
    )
    assert cmp.time.equals(expected_time)
    assert np.all(cmp.x == np.array([3.0, 4.0]))


def test_tiny_mod_unique(obs_tiny, mod_tiny_unique):
    cmp = ms.match(obs_tiny, mod_tiny_unique)
    assert cmp.n_points == 4
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:04",
        ]
    )
    assert cmp.time.equals(expected_time)
    assert np.all(cmp.x == np.array([1.0, 2.0, 3.0, 4.0]))


# Currently fails as check on x, y difference is missing!
def test_tiny_mod_xy_difference(obs_tiny_df, mod_tiny_unique):
    obs_tiny_df.loc["2017-10-27 13:00:01", "x"] = (
        1.1  # difference in x larger than tolerance
    )
    obs_tiny_df.loc["2017-10-27 13:00:03", "y"] = (
        13.6  # difference in y larger than tolerance
    )
    with pytest.warns(UserWarning, match="Removed 2 duplicate timestamps"):
        obs_tiny = ms.TrackObservation(
            obs_tiny_df, item="alti", x_item="x", y_item="y", keep_duplicates="first"
        )
    with pytest.warns(UserWarning, match="Removed 2 model points"):
        # 2 points removed due to difference in x,y
        cmp = ms.match(obs_tiny, mod_tiny_unique)
    assert cmp.n_points == 2
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:04",
        ]
    )
    assert cmp.time.equals(expected_time)
    assert np.all(cmp.x == np.array([2.0, 4.0]))


def test_tiny_mod_rounding_error(obs_tiny, mod_tiny_rounding_error):
    # accepts rounding error in x, y
    cmp = ms.match(obs_tiny, mod_tiny_rounding_error)
    assert cmp.n_points == 4
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:04",
        ]
    )
    assert cmp.time.equals(expected_time)
    assert np.all(cmp.x == np.array([1.0, 2.0, 3.0, 4.0]))


@pytest.fixture
def observation_df():
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    return pd.read_csv(fn, index_col=0, parse_dates=True)


@pytest.fixture
def observation(observation_df):
    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        o = ms.TrackObservation(observation_df, item=2, name="alti")
    return o


@pytest.fixture
def modelresult():
    fn = "tests/testdata/NorthSeaHD_extracted_track.dfs0"
    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        mr = ms.model_result(fn, gtype="track", item=2, name="HD")
    return mr


@pytest.fixture
def comparer(observation, modelresult):
    return ms.match(observation, modelresult)


def test_skill(comparer):
    c = comparer
    df = c.skill().to_dataframe()

    # assert df.loc["alti"].n == 532  # 544
    assert df[0, "n"] == 532  # 544


# def test_extract_no_time_overlap(modelresult, observation_df):
#     mr = modelresult
#     df = observation_df.copy(deep=True)
#     df.index = df.index + np.timedelta64(100, "D")
#     with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
#         o = ms.TrackObservation(df, item=2, name="alti")

#     with pytest.raises(ValueError, match="Validation failed"):
#         with pytest.warns(UserWarning, match="No time overlap!"):
#             ms.Connector(o, mr)

#     with pytest.warns(UserWarning, match="No time overlap!"):
#         con = ms.Connector(o, mr, validate=False)

#     with pytest.warns(UserWarning, match="No overlapping data"):
#         cc = con.extract()

#     assert cc.n_comparers == 0


# def test_extract_obs_start_before(modelresult, observation_df):
#     mr = modelresult
#     df = observation_df.copy(deep=True)
#     df.index = df.index - np.timedelta64(1, "D")
#     with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
#         o = ms.TrackObservation(df, item=2, name="alti")
#     con = ms.Connector(o, mr)
#     with pytest.warns(UserWarning, match="No overlapping data"):
#         cc = con.extract()
#     assert cc.n_comparers == 0


# def test_extract_obs_end_after(modelresult, observation_df):
#     mr = modelresult
#     df = observation_df.copy(deep=True)
#     df.index = df.index + np.timedelta64(1, "D")
#     with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
#         o = ms.TrackObservation(df, item=2, name="alti")
#     con = ms.Connector(o, mr)
#     with pytest.warns(UserWarning, match="No overlapping data"):
#         cc = con.extract()
#     assert cc.n_comparers == 0


# def test_extract_no_spatial_overlap_dfs0(modelresult, observation_df):
#     mr = modelresult
#     df = observation_df.copy(deep=True)
#     df.lon = -100
#     df.lat = -50
#     # with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
#     o = ms.TrackObservation(df, item=2, name="alti")
#     con = ms.Connector(o, mr)
#     with pytest.warns(UserWarning, match="No overlapping data"):
#     cc = con.extract()

# assert cc.n_comparers == 0
# assert cc.n_points == 0


# def test_extract_no_spatial_overlap_dfsu(observation_df):


def test_skill_vs_gridded_skill(comparer):
    import polars as pl

    df = comparer.skill().to_dataframe()  # to compare to result of .skill()
    ds = comparer.gridded_skill(bins=1)  # force 1 bin only

    row = df.filter(pl.col("observation") == "alti")
    assert row[0, "n"] == ds.data.n.values
    assert row[0, "bias"] == pytest.approx(ds.data.bias.values)
    # assert df.loc["alti"].n == ds.data.n.values
    # assert df.loc["alti"].bias == ds.data.bias.values
    assert ds.x.size == 1
    assert ds.y.size == 1

    # assert ds.coords._names == {"x","y"}  # TODO: Why return "observation" as by, when n_obs==1 but not "model"?


def test_gridded_skill_bins(comparer: ms.Comparer) -> None:
    # default
    ds = comparer.gridded_skill(metrics=["bias"])
    assert len(ds.x) == 5
    assert len(ds.y) == 5

    # float
    ds = comparer.gridded_skill(metrics=["bias"], bins=2)
    assert len(ds.x) == 2
    assert len(ds.y) == 2

    # float for x and range for y
    ds = comparer.gridded_skill(metrics=["bias"], bins=(2, [50, 50.5, 51, 53]))
    assert len(ds.x) == 2
    assert len(ds.y) == 3

    # binsize (overwrites bins)
    ds = comparer.gridded_skill(metrics=["bias"], binsize=2.5, bins=100)
    assert len(ds.x) == 4
    assert len(ds.y) == 3
    assert ds.x[0] == -0.75


# This test doesn't test anything meaningful
# def test_gridded_skill_by(comparer):
#     ds = comparer.gridded_skill(metrics=["bias"], by=["y", "mod"])
#     assert ds.coords._names == {"y", "model", "x"}


def test_gridded_skill_misc(comparer: ms.Comparer) -> None:
    # miniumum n
    # ds = comparer.gridded_skill(metrics=["bias", "rmse"], n_min=20)
    ds = comparer.gridded_skill(metrics=["bias", "rmse"], n_min=20)
    df = ds.to_dataframe()
    # assert df.loc[df.n < 20, ["bias", "rmse"]].size == 30
    assert df.loc[df.n < 20, ["bias", "rmse"]].isna().all().all()


def test_hist(comparer):
    cmp = comparer

    cmp.plot.hist(bins=np.linspace(0, 7, num=15))

    cmp.plot.hist(bins=10)
    cmp.plot.hist(density=False)
    cmp.sel(model=0).plot.hist(title="new_title", alpha=0.2)


def test_residual_hist(comparer):
    cmp = comparer
    cmp.plot.residual_hist()
    cmp.plot.residual_hist(bins=10, title="new_title", color="blue")


def test_df_input(obs_tiny_df, mod_tiny3):
    """A dataframe is a valid input to ms.match, without explicitly creating a TrackObservation"""
    # excerpt from obs_tiny_df
    # time                | value
    # --------------------------
    # 2017-10-27 13:00:02  2.0
    # 2017-10-27 13:00:02  3.0

    assert isinstance(obs_tiny_df, pd.DataFrame)
    assert len(obs_tiny_df["2017-10-27 13:00:02":"2017-10-27 13:00:02"]) == 2

    with pytest.warns(UserWarning, match="Removed 2 duplicate timestamps"):
        cmp = ms.match(obs_tiny_df, mod_tiny3, gtype="track")

    assert (
        cmp.data.sel(
            time=slice("2017-10-27 13:00:02", "2017-10-27 13:00:02")
        ).Observation
        == 2.0  # first value
    )
