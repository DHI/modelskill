import pytest

from fmskill.model import ModelResult, ModelResultCollection
from fmskill.observation import PointObservation
from fmskill.metrics import root_mean_squared_error, mean_absolute_error


@pytest.fixture
def mr1():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    return ModelResult(fn, name="SW_1")


@pytest.fixture
def mr2():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    return ModelResult(fn, name="SW_2")


@pytest.fixture
def o1():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def o2():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def mrc(mr1, mr2, o1, o2):
    return ModelResultCollection([mr1, mr2])


@pytest.fixture
def mrc_o1o2(mr1, mr2, o1, o2):
    mrc = ModelResultCollection([mr1, mr2])
    mrc.add_observation(o1, item=0)
    mrc.add_observation(o2, item=0)
    return mrc


def test_mrc_repr(mrc):
    txt = repr(mrc)
    assert "ModelResultCollection" in txt


def test_add_observation(mrc, o1):
    mrc.add_observation(o1, item=0)
    assert len(mrc.observations) == 1


def test_compare_point_observation(mrc, o1):
    mrc.compare_point_observation(o1, item=0)
    assert True


def test_extract(mrc_o1o2):
    cc = mrc_o1o2.extract()
    assert True
