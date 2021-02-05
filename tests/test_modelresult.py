import pytest

from mikefm_skill.model import ModelResult, ModelResultType
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
def hd_oresund_2d():
    return "tests/testdata/Oresund2D.dfsu"


@pytest.fixture
def Hm0_HKNA():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def Hm0_EPL():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def sw_dutch_coast():
    return "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"


def test_dfs_object(hd_oresund_2d):
    mr = ModelResult(hd_oresund_2d)

    assert mr.dfs.is_2d


def test_ModelResultType(sw_dutch_coast):
    mr = ModelResult(sw_dutch_coast)

    assert mr.type == ModelResultType.dfsu


def test_ModelResultType0():
    mr = ModelResult("tests/testdata/TS.dfs0")

    assert mr.type == ModelResultType.dfs0


def test_extract(hd_oresund_2d, klagshamn, drogden):
    mr = ModelResult(hd_oresund_2d)

    mr.add_observation(klagshamn, item=0)
    mr.add_observation(drogden, item=0)
    collection = mr.extract()
    collection[0].name == "Klagshamn"


def test_plot_positions(sw_dutch_coast, Hm0_EPL, Hm0_HKNA):
    mr = ModelResult(sw_dutch_coast)
    mr.add_observation(Hm0_EPL, item=0)
    mr.add_observation(Hm0_HKNA, item=0)
    mr.plot_observation_positions()
