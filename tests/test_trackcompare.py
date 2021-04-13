import pytest
import pandas as pd
import numpy as np

from fmskill.model import ModelResult
from fmskill.observation import TrackObservation


@pytest.fixture
def observation():
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    df = pd.read_csv(fn, index_col=0, parse_dates=True)
    return TrackObservation(df, item=2, name="alti")


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


def test_skill_vs_spatial_skill(comparer):
    df = comparer.skill()  # to compare to result of .skill()
    ds = comparer.spatial_skill()

    assert df.loc["alti"].n == ds.n.values.sum()
    assert ds.xBin.__len__() == 5
    assert ds.yBin.__len__() == 5
    # assert ds.coords._names == {"xBin","yBin"}  # TODO: Why return "observation" a by, when n_obs==1?


def test_spatial_skill_bins(comparer):
    # default
    ds = comparer.spatial_skill(metrics=["bias"])
    assert ds.xBin.__len__() == 5
    assert ds.yBin.__len__() == 5

    # float
    ds = comparer.spatial_skill(metrics=["bias"], bins=2)
    assert ds.xBin.__len__() == 2
    assert ds.yBin.__len__() == 2

    # float for x and range for y
    ds = comparer.spatial_skill(metrics=["bias"], bins=(2, [50, 50.5, 51, 53]))
    assert ds.xBin.__len__() == 2
    assert ds.yBin.__len__() == 3

    # binsize (overwrites bins)
    ds = comparer.spatial_skill(metrics=["bias"], binsize=2.5, bins=100)
    assert ds.xBin.__len__() == 4
    assert ds.yBin.__len__() == 3
    assert ds.xBin[0] == -0.75


def test_spatial_skill_by(comparer):
    # odd order of by
    ds = comparer.spatial_skill(metrics=["bias"], by=["yBin", "mod"])
    assert ds.coords._names == {"xBin", "model", "yBin"}


def test_spatial_skill_misc(comparer):
    # miniumum n
    ds = comparer.spatial_skill(metrics=["bias", "rmse"], n_min=20)
    df = ds.to_dataframe()
    assert df.loc[df.n < 20, ["bias", "rmse"]].size
