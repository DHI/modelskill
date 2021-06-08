from datetime import datetime
import os
import warnings

from fmskill.data.dmi import DMIOceanObsRepository


def test_get_observations():

    if "DMI_API_KEY" not in os.environ:
        warnings.warn("Environment variable 'DMI_API_KEY' not present, skipping test")

    api_key = os.environ["DMI_API_KEY"]
    repo = DMIOceanObsRepository(apikey=api_key)

    station_id = "30336"  # Kbh havn

    df = repo.get_observations(
        station_id=station_id, start_time=datetime(2020, 1, 1), n=100
    )

    assert df.shape[0] > 0