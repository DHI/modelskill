import pytest
import pandas as pd
import numpy as np

from modelskill import ModelResult
from modelskill import TrackObservation
from modelskill import Connector


@pytest.fixture
def observation_df():
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    return pd.read_csv(fn, index_col=0, parse_dates=True)


@pytest.fixture
def observation(observation_df):
    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        o = TrackObservation(observation_df, item=2, name="alti")
    return o


@pytest.fixture
def modelresult():
    fn = "tests/testdata/NorthSeaHD_extracted_track.dfs0"
    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        mr = ModelResult(fn, gtype="track", item=2, name="HD")
    return mr


@pytest.fixture
def comparer(observation, modelresult):
    con = Connector(observation, modelresult)
    cc = con.extract()
    return cc


def test_skill(comparer):
    c = comparer
    df = c.skill().df

    assert df.loc["alti"].n == 544


def test_extract_no_time_overlap(modelresult, observation_df):
    mr = modelresult
    df = observation_df.copy(deep=True)
    df.index = df.index + np.timedelta64(100, "D")
    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        o = TrackObservation(df, item=2, name="alti")

    with pytest.raises(ValueError, match="Validation failed"):
        with pytest.warns(UserWarning, match="No time overlap!"):
            Connector(o, mr)

    with pytest.warns(UserWarning, match="No time overlap!"):
        con = Connector(o, mr, validate=False)

    with pytest.warns(UserWarning, match="No overlapping data"):
        cc = con.extract()

    assert cc.n_comparers == 0


def test_extract_obs_start_before(modelresult, observation_df):
    mr = modelresult
    df = observation_df.copy(deep=True)
    df.index = df.index - np.timedelta64(1, "D")
    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        o = TrackObservation(df, item=2, name="alti")
    con = Connector(o, mr)
    with pytest.warns(UserWarning, match="No overlapping data"):
        cc = con.extract()
    assert cc.n_comparers == 0


def test_extract_obs_end_after(modelresult, observation_df):
    mr = modelresult
    df = observation_df.copy(deep=True)
    df.index = df.index + np.timedelta64(1, "D")
    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        o = TrackObservation(df, item=2, name="alti")
    con = Connector(o, mr)
    with pytest.warns(UserWarning, match="No overlapping data"):
        cc = con.extract()
    assert cc.n_comparers == 0


def test_extract_no_spatial_overlap_dfs0(modelresult, observation_df):
    mr = modelresult
    df = observation_df.copy(deep=True)
    df.lon = -100
    df.lat = -50
    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        o = TrackObservation(df, item=2, name="alti")
    con = Connector(o, mr)
    with pytest.warns(UserWarning, match="No overlapping data"):
        cc = con.extract()

    assert cc.n_comparers == 0
    assert cc.n_points == 0


# def test_extract_no_spatial_overlap_dfsu(observation_df):


def test_skill_vs_spatial_skill(comparer):
    df = comparer.skill().df  # to compare to result of .skill()
    ds = comparer.spatial_skill(bins=1)  # force 1 bin only

    assert df.loc["alti"].n == ds.n.values
    assert df.loc["alti"].bias == ds.ds.bias.values
    assert ds.x.size == 1
    assert ds.y.size == 1
    # assert ds.coords._names == {"x","y"}  # TODO: Why return "observation" as by, when n_obs==1 but not "model"?


def test_spatial_skill_bins(comparer):
    # default
    ds = comparer.spatial_skill(metrics=["bias"])
    assert len(ds.x) == 5
    assert len(ds.y) == 5

    # float
    ds = comparer.spatial_skill(metrics=["bias"], bins=2)
    assert len(ds.x) == 2
    assert len(ds.y) == 2

    # float for x and range for y
    ds = comparer.spatial_skill(metrics=["bias"], bins=(2, [50, 50.5, 51, 53]))
    assert len(ds.x) == 2
    assert len(ds.y) == 3

    # binsize (overwrites bins)
    ds = comparer.spatial_skill(metrics=["bias"], binsize=2.5, bins=100)
    assert len(ds.x) == 4
    assert len(ds.y) == 3
    assert ds.x[0] == -0.75


def test_spatial_skill_by(comparer):
    # odd order of by
    ds = comparer.spatial_skill(metrics=["bias"], by=["y", "mod"])
    assert ds.coords._names == {"y", "model", "x"}


def test_spatial_skill_misc(comparer):
    # miniumum n
    ds = comparer.spatial_skill(metrics=["bias", "rmse"], n_min=20)
    df = ds.to_dataframe()
    assert df.loc[df.n < 20, ["bias", "rmse"]].size == 30
    assert df.loc[df.n < 20, ["bias", "rmse"]].isna().all().all()


def test_hist(comparer):
    cc = comparer

    with pytest.warns(FutureWarning):
        cc.hist()

    cc.plot.hist(bins=np.linspace(0, 7, num=15))

    cc[0].plot.hist(bins=10)
    cc[0].plot.hist(density=False)
    cc[0].plot.hist(model=0, title="new_title", alpha=0.2)


def test_residual_hist(comparer):
    cc = comparer
    cc[0].plot.residual_hist()
    cc[0].plot.residual_hist(bins=10, title="new_title", color="blue")
