import pytest

from shapely.geometry import Point
from mikefm_skill.observation import PointObservation

@pytest.fixture
def klagshamn():
    return "tests/testdata/smhi_2095_klagshamn.dfs0"

def test_coordinates(klagshamn):
    o1 = PointObservation(klagshamn,item=0,x=0.36684415E+06,y=0.61542916E+07) #lon=12.89106996, lat=55.5165157
    assert isinstance(o1.geo, Point)