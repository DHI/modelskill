from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import pytest
from fmskill.data import DHIAltimetryRepository
from fmskill.data.altimetry import AltimetryData
from mikeio import Dfs0, eum

try:
    import geopandas as gpd
except ImportError:
    pytest.skip("geopandas not available", allow_module_level=True)


def requires_DHI_ALTIMETRY_API_KEY():
    api_key = os.environ.get("DHI_ALTIMETRY_API_KEY")
    reason = "Environment variable DHI_ALTIMETRY_API_KEY not present"
    return pytest.mark.skipif(api_key is None, reason=reason)


@pytest.fixture
def repo():
    api_key = os.environ["DHI_ALTIMETRY_API_KEY"]
    return DHIAltimetryRepository(api_key=api_key)


@pytest.fixture
def ad85(repo):
    area = "lon=10.9&lat=55.9&radius=20"
    start = "1985"
    end = "1985-5-1"
    return repo.get_altimetry_data(area=area, start_time=start, end_time=end)


@pytest.fixture
def ad(repo):
    area = "bbox=10,55,11.5,56.5"
    start = "2020-1-1"
    end = "2020-1-7"
    return repo.get_altimetry_data(area=area, start_time=start, end_time=end)


@requires_DHI_ALTIMETRY_API_KEY()
def test_get_satellites(repo):
    sats = repo.satellites
    assert "3a" in sats


@requires_DHI_ALTIMETRY_API_KEY()
def test_quality_filters(repo):
    qf = repo.get_quality_filters()
    assert "qual_wind_speed" in qf.index


@requires_DHI_ALTIMETRY_API_KEY()
def test_get_daily_count(repo):
    area = "lon=10.9&lat=55.9&radius=10.0"
    df = repo.get_daily_count(area, start_time="2021")
    assert df.loc["2021-1-4"].values == 4


@requires_DHI_ALTIMETRY_API_KEY()
def test_parse_satellites(repo):
    sats = repo.parse_satellites("3b")
    assert sats[0] == "3b"


@requires_DHI_ALTIMETRY_API_KEY()
def test_time_of_newest_data(repo):
    latest = repo.time_of_newest_data()
    day_before_yesterday = datetime.now() - timedelta(days=2)
    assert latest > day_before_yesterday


@requires_DHI_ALTIMETRY_API_KEY()
def test_plot_observation_stats(repo):
    repo.plot_observation_stats()
    assert True


@requires_DHI_ALTIMETRY_API_KEY()
def test_get_spatial_coverage(repo):
    gdf = repo.get_spatial_coverage(
        area="lon=10.9&lat=55.9&radius=40", start_time="2021-1-1", end_time="2021-1-5"
    )
    assert gdf[["count"]].loc[0].values[0] == 8
    # assert isinstance(gdf, gpd.GeoDataFrame)
    gdf.plot("count")
    assert True


@requires_DHI_ALTIMETRY_API_KEY()
def test_get_altimetry_data_1985(ad85):
    assert isinstance(ad85, AltimetryData)
    assert ad85.df.index.is_unique
    row = ad85.df.iloc[0]
    assert row.lon == 10.795129
    assert np.isnan(row.water_level)
    assert row.wind_speed == 2.2


@requires_DHI_ALTIMETRY_API_KEY()
def test_get_altimetry_data_no_end(repo):
    area = "lon=10.9&lat=55.9&radius=50"
    start = datetime.now() - timedelta(days=7)  # latest 7days

    ad = repo.get_altimetry_data(area=area, start_time=start)
    assert isinstance(ad, AltimetryData)
    assert ad.df.index.is_unique
    assert ad.n_points > 0


@requires_DHI_ALTIMETRY_API_KEY()
def test_AltimetryData_properties(ad):
    assert len(ad.satellites) == 5
    assert "3b" in ad.satellites
    assert datetime.date(ad.start_time) == pd.to_datetime(ad.query_params["start_date"])
    assert datetime.date(ad.end_time) + timedelta(days=1) == pd.to_datetime(
        ad.query_params["end_date"]
    )


@requires_DHI_ALTIMETRY_API_KEY()
def test_AltimetryData_to_dfs0(ad, tmpdir):

    outfilename = os.path.join(tmpdir, "out.dfs0")
    ad.to_dfs0(outfilename)
    assert os.path.exists(outfilename)
    dfs = Dfs0(outfilename)
    assert dfs.items[0].type == eum.EUMType.Latitude_longitude
    assert dfs.items[2].type == eum.EUMType.Water_Level


@requires_DHI_ALTIMETRY_API_KEY()
def test_AltimetryData_plot_map(ad):
    ad.plot_map()


@requires_DHI_ALTIMETRY_API_KEY()
def test_AltimetryData_assign_track_id(ad):
    df = ad.assign_track_id()
    assert "track_id" in df.columns
    assert df.track_id.to_numpy()[0] == 0


@requires_DHI_ALTIMETRY_API_KEY()
def test_AltimetryData_print_records_per_satellite(ad):
    ad.print_records_per_satellite()
    assert True