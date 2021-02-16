import pytest

from mikefm_skill.model import ModelResult
from mikefm_skill.observation import PointObservation
from mikefm_skill.metrics import root_mean_squared_error, mean_absolute_error


@pytest.fixture
def klagshamn():
    fn = "tests/testdata/smhi_2095_klagshamn.dfs0"
    return PointObservation(fn, item=0, x=366844, y=6154291, name="Klagshamn")


@pytest.fixture
def drogden():
    fn = "tests/testdata/dmi_30357_Drogden_Fyr.dfs0"
    return PointObservation(fn, item=0, x=355568.0, y=6156863.0)


@pytest.fixture
def modelresult_oresund_2d():
    return ModelResult("tests/testdata/Oresund2D.dfsu")


def test_compound_skill(modelresult_oresund_2d, klagshamn, drogden):
    mr = modelresult_oresund_2d

    mr.add_observation(klagshamn, item=0)
    mr.add_observation(drogden, item=0)
    collection = mr.extract()

    assert collection.compound_skill(metric=root_mean_squared_error) > 0.0
    report = collection.skill_report(
        metrics=[root_mean_squared_error, mean_absolute_error]
    )


def test_compound_weighted_skill(modelresult_oresund_2d, klagshamn, drogden):
    mr = modelresult_oresund_2d

    mr.add_observation(klagshamn, item=0)
    mr.add_observation(drogden, item=0)
    c = mr.extract()
    unweighted_skill = c.compound_skill()

    mrw = modelresult_oresund_2d

    mrw.add_observation(klagshamn, item=0, weight=1.0)
    mrw.add_observation(drogden, item=0, weight=0.0)
    cw = mrw.extract()

    weighted_skill = cw.compound_skill()

    assert unweighted_skill != weighted_skill
