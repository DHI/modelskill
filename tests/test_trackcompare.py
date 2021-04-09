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
    # default should return same result as .skill() but as dataset
    df = comparer.skill(metrics=["rmse", "bias"])
    ds = comparer.spatial_skill(metrics=["rmse", "bias"])

    assert df.loc["alti"].n == ds.n.values.sum()
    # ds.sel(mod_name="HD", obs_name="alti").n
    # assert df.loc["alti"].rmse == ds.sel(mod_name="HD", obs_name="alti").rmse
