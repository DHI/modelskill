import pytest
import numpy as np

from fmskill.model import ModelResult
from fmskill.observation import PointObservation
from fmskill.metrics import root_mean_squared_error, mean_absolute_error


@pytest.fixture
def klagshamn():
    fn = "tests/testdata/smhi_2095_klagshamn.dfs0"
    return PointObservation(
        fn, item=0, x=366844, y=6154291, name="Klagshamn", variable_name="WL"
    )


@pytest.fixture
def drogden():
    fn = "tests/testdata/dmi_30357_Drogden_Fyr.dfs0"
    return PointObservation(fn, item=0, x=355568.0, y=6156863.0, variable_name="WL")


@pytest.fixture
def modelresult_oresund_2d():
    return ModelResult("tests/testdata/Oresund2D.dfsu")


def test_skill_from_observation_with_missing_values(modelresult_oresund_2d):
    o1 = PointObservation(
        "tests/testdata/eq_ts_with_gaps.dfs0",
        item=0,
        x=366844,
        y=6154291,
        name="Klagshamn",
    )
    mr = modelresult_oresund_2d
    mr.add_observation(o1, item=0)
    c = mr.extract()
    s = c["Klagshamn"].skill()
    assert not np.any(np.isnan(s))


def test_score(modelresult_oresund_2d, klagshamn, drogden):
    mr = modelresult_oresund_2d

    mr.add_observation(klagshamn, item=0, validate_eum=False)
    mr.add_observation(drogden, item=0, validate_eum=False)
    collection = mr.extract()

    assert collection.score(metric=root_mean_squared_error) > 0.0
    report = collection.skill(metrics=[root_mean_squared_error, mean_absolute_error])


def test_weighted_score(modelresult_oresund_2d, klagshamn, drogden):
    mr = modelresult_oresund_2d

    mr.add_observation(klagshamn, item=0, validate_eum=False)
    mr.add_observation(drogden, item=0, validate_eum=False)
    c = mr.extract()
    unweighted_skill = c.score()

    mrw = modelresult_oresund_2d

    mrw.add_observation(klagshamn, item=0, weight=1.0, validate_eum=False)
    mrw.add_observation(drogden, item=0, weight=0.0, validate_eum=False)
    cw = mrw.extract()

    weighted_skill = cw.score()

    assert unweighted_skill != weighted_skill


def test_misc_properties(klagshamn, drogden):

    mr = ModelResult("tests/testdata/Oresund2D.dfsu")

    mr.add_observation(klagshamn, item=0, validate_eum=False)
    mr.add_observation(drogden, item=0, validate_eum=False)

    c = mr.extract()

    assert len(c) == 2
    assert c.n_comparers == 2

    assert c.n_models == 1
    assert c._mod_names == [
        "Oresund2D"
    ]  # TODO this fails when all tests are run, something is spilling over from another test !!

    ck = c["Klagshamn"]
    assert ck.name == "Klagshamn"

    assert ck.n_points > 0

    assert ck.start.year == 2018  # intersection of observation and model times
    assert ck.end.year == 2018

    assert ck.x == 366844


def test_skill(klagshamn, drogden):

    mr = ModelResult("tests/testdata/Oresund2D.dfsu")

    mr.add_observation(klagshamn, item=0, validate_eum=False)
    mr.add_observation(drogden, item=0, validate_eum=False)

    c = mr.extract()

    df = c.skill()
    assert df.loc["Klagshamn"].n == 167

    # Filtered skill
    df = c.skill(observation="Klagshamn")
    assert df.loc["Klagshamn"].n == 167
