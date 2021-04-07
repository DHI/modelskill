import pytest

from fmskill.model import ModelResult, ModelResultCollection
from fmskill.observation import PointObservation, TrackObservation
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
def o3():
    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    return TrackObservation(fn, item=3, name="c2")


@pytest.fixture
def mrc(mr1, mr2):
    return ModelResultCollection([mr1, mr2])


@pytest.fixture
def mrc_o1o2(mr1, mr2, o1, o2):
    mrc = ModelResultCollection([mr1, mr2])
    mrc.add_observation(o1, item=0)
    mrc.add_observation(o2, item=0)
    return mrc


@pytest.fixture
def cc(mr1, mr2, o1, o2, o3):
    mrc = ModelResultCollection([mr1, mr2])
    mrc.add_observation(o1, item=0)
    mrc.add_observation(o2, item=0)
    mrc.add_observation(o3, item=0)
    return mrc.extract()


def test_mrc_repr(mrc):
    txt = repr(mrc)
    assert "ModelResultCollection" in txt


def test_add_observation(mrc, o1):
    mrc.add_observation(o1, item=0)
    assert len(mrc.observations) == 1


# def test_compare_point_observation(mrc, o1):
#    mrc.compare_point_observation(o1, item=0)
#    assert True


def test_extract(mrc_o1o2):
    cc = mrc_o1o2.extract()
    assert True


def test_mm_skill(cc):
    df = cc.skill(model="SW_1")
    assert df.loc["EPL"].n == 66
    assert df.loc["c2"].n == 113

