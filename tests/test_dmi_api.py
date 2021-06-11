from datetime import datetime
import os

from fmskill.data.dmi import DMIOceanObsRepository


def test_get_observed_data():

    api_key = os.environ["DMI_API_KEY"]
    repo = DMIOceanObsRepository(apikey=api_key)

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


def test_get_stations():
    api_key = os.environ["DMI_API_KEY"]
    repo = DMIOceanObsRepository(apikey=api_key)

    assert repo.stations.shape[0] > 0

    cols = repo.stations.columns

    assert "station_id" in cols
    assert "lon" in cols
    assert "lat" in cols
    assert "name" in cols
