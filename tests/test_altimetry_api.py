from datetime import datetime
import os
import pytest
from fmskill.data import DHIAltimetryRepository


def requires_DHI_ALTIMETRY_API_KEY():
    api_key = os.environ.get("DHI_ALTIMETRY_API_KEY")
    reason = "Environment variable DHI_ALTIMETRY_API_KEY not present"
    return pytest.mark.skipif(api_key is None, reason=reason)


@pytest.fixture
def repo():
    api_key = os.environ["DHI_ALTIMETRY_API_KEY"]
    return DHIAltimetryRepository(api_key=api_key)


@requires_DHI_ALTIMETRY_API_KEY()
def test_get_satellites(repo):
    sats = repo.satellites
    assert "3a" in sats
