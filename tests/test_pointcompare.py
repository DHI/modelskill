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
def oresund_2d():
    return "tests/testdata/Oresund2D.dfsu"


def test_dfs_object(oresund_2d):
    mr = ModelResult(oresund_2d)

    assert mr.dfs.is_2d


def test_total_skill(oresund_2d, klagshamn, drogden):
    mr = ModelResult(oresund_2d)

    mr.add_observation(klagshamn, item=0)
    mr.add_observation(drogden, item=0)
    collection = mr.extract()

    assert collection.skill(metric=root_mean_squared_error) > 0.0
    report = collection.skill_report(
        metrics=[root_mean_squared_error, mean_absolute_error]
    )

    print(report)

