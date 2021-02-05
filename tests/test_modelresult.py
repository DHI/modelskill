import pytest

from mikefm_skill.model import ModelResult
from mikefm_skill.observation import PointObservation


@pytest.fixture
def klagshamn():
    fn = "tests/testdata/smhi_2095_klagshamn.dfs0"
    return PointObservation(fn, item=0, x=366844, y=6154291, name="Klagshamn")


@pytest.fixture
def drogden():

    # >>> from pyproj import Transformer
    # >>> t = Transformer.from_crs(4326,32633, always_xy=True)
    # >>> t.transform(12.7113,55.5364)
    # (355568.6130331255, 6156863.0187071245)

    fn = "tests/testdata/dmi_30357_Drogden_Fyr.dfs0"
    return PointObservation(fn, item=0, x=355568.0, y=6156863.0)


@pytest.fixture
def oresund_2d():
    return "tests/testdata/Oresund2D.dfsu"


def test_dfs_object(oresund_2d):
    mr = ModelResult(oresund_2d)

    assert mr.dfs.is_2d


def test_extract(oresund_2d, klagshamn, drogden):
    mr = ModelResult(oresund_2d)

    mr.add_observation(klagshamn, item=0)
    mr.add_observation(drogden, item=0)
    collection = mr.extract()
    collection[0].name == "Klagshamn"

