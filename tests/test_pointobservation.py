import pytest

from shapely.geometry import Point
from mikefm_skill.observation import PointObservation


@pytest.fixture
def klagshamn():
    return "tests/testdata/smhi_2095_klagshamn.dfs0"


def test_coordinates(klagshamn):
    o1 = PointObservation(klagshamn, item=0, x=366844, y=6154291, name="Klagshamn")
    assert isinstance(o1.geometry, Point)
