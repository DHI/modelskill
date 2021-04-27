import pytest
import pandas as pd
import numpy as np

from fmskill.model import ModelResult
from fmskill.observation import TrackObservation


@pytest.fixture
def observation_df():
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    return pd.read_csv(fn, index_col=0, parse_dates=True)


@pytest.fixture
def observation(observation_df):
    return TrackObservation(observation_df, item=2, name="alti")


@pytest.fixture
def modelresult():
    fn = "tests/testdata/NorthSeaHD_extracted_track.dfs0"
    return ModelResult(fn, name="HD")


@pytest.fixture
def comparer(observation, modelresult):
    mr = modelresult
    mr.add_observation(observation, item=2)
    return mr.extract()


def test_skill(comparer):
    c = comparer
    df = c.skill()

    assert df.loc["alti"].n == 544


def test_extract_no_time_overlap(modelresult, observation_df):
    mr = modelresult
    df = observation_df.copy(deep=True)
    df.index = df.index + np.timedelta64(100, "D")
    o = TrackObservation(df, item=2, name="alti")
    mr.add_observation(o, item=2)
    cc = mr.extract()


def test_extract_obs_start_before(modelresult, observation_df):
    mr = modelresult
    df = observation_df.copy(deep=True)
    df.index = df.index - np.timedelta64(1, "D")
    o = TrackObservation(df, item=2, name="alti")
    mr.add_observation(o, item=2)
    cc = mr.extract()


def test_extract_obs_end_after(modelresult, observation_df):
    mr = modelresult
    df = observation_df.copy(deep=True)
    df.index = df.index + np.timedelta64(1, "D")
    o = TrackObservation(df, item=2, name="alti")
    mr.add_observation(o, item=2)
    cc = mr.extract()


def test_extract_no_spatial_overlap_dfs0(modelresult, observation_df):
    mr = modelresult
    df = observation_df.copy(deep=True)
    df.lon = -100
    df.lat = -50
    o = TrackObservation(df, item=2, name="alti")
    mr.add_observation(o, item=2)
    cc = mr.extract()

    assert len(cc.all_df) == 0


# def test_extract_no_spatial_overlap_dfsu(observation_df):


def test_skill_vs_spatial_skill(comparer):
    df = comparer.skill()  # to compare to result of .skill()
    ds = comparer.spatial_skill(bins=1)  # force 1 bin only

    assert df.loc["alti"].n == ds.n.values
    assert df.loc["alti"].bias == ds.bias.values
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
