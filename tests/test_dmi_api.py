from datetime import datetime
import os
import pandas as pd
import pytest
from fmskill.data.dmi import DMIOceanObsRepository


def requires_DMI_API_KEY():
    api_key = os.environ.get("DMI_API_KEY")
    reason = "Environment variable DMI_API_KEY not present"
    return pytest.mark.skipif(api_key is None, reason=reason)


@pytest.fixture
def repo():
    api_key = os.environ["DMI_API_KEY"]
    return DMIOceanObsRepository(api_key=api_key)


@requires_DMI_API_KEY()
def test_get_observed_data(repo):

    station_id = "30336"  # Kbh havn

    df = repo.get_observed_data(
        station_id=station_id,
        start_time=datetime(2020, 1, 1),
        end_time=datetime(2020, 1, 2),
        limit=10,
    )

    assert df.shape[0] > 0
    assert df.index[0].year == 2020
    assert df.index[-1].year == 2020


@requires_DMI_API_KEY()
def test_get_observed_data_future_no_data(repo):

    df = repo.get_observed_data(
        station_id="31623",
        start_time=datetime(2100, 1, 1),  # in the future
    )

    assert df.shape[0] == 0


@requires_DMI_API_KEY()
def test_get_observed_data_concatenatable(repo):

    station_id = "31616"
    parameter_id = "sealev_dvr"

    df1 = repo.get_observed_data(
        station_id=station_id,
        parameter_id=parameter_id,
        start_time=datetime(2020, 1, 1),
        end_time=datetime(2020, 1, 2),
    )

    df2 = repo.get_observed_data(
        station_id=station_id,
        parameter_id=parameter_id,
        start_time=datetime(2021, 1, 1),
        end_time=datetime(2021, 1, 2),
    )

    dff = repo.get_observed_data(
        station_id=station_id,
        parameter_id=parameter_id,
        start_time=datetime(2100, 1, 1),  # in the future
    )

    df = pd.concat([df1, df2, dff])

    assert parameter_id in df.columns
    assert df.shape[0] > df1.shape[0]
    assert df.shape[0] > df2.shape[0]
    assert df1.index.min() == df.index.min()


@requires_DMI_API_KEY()
def test_get_stations(repo):

    assert repo.stations.shape[0] > 0

    cols = repo.stations.columns

    assert "station_id" in cols
    assert "lon" in cols
    assert "lat" in cols
    assert "name" in cols
