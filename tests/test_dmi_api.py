from datetime import datetime
import os
import pytest
from fmskill.data.dmi import DMIOceanObsRepository


def requires_DMI_API_KEY():
    api_key = os.environ.get("DMI_API_KEY")
    reason = "Environment variable DMI_API_KEY not present"
    return pytest.mark.skipif(api_key is None, reason=reason)


@requires_DMI_API_KEY()
def test_get_observed_data():

    api_key = os.environ["DMI_API_KEY"]
    repo = DMIOceanObsRepository(api_key=api_key)

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
def test_get_stations():
    api_key = os.environ["DMI_API_KEY"]
    repo = DMIOceanObsRepository(api_key=api_key)

    assert repo.stations.shape[0] > 0

    cols = repo.stations.columns

    assert "station_id" in cols
    assert "lon" in cols
    assert "lat" in cols
    assert "name" in cols
