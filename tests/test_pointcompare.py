import pytest
import numpy as np

from fmskill.model import ModelResult
from fmskill.observation import PointObservation
from fmskill.metrics import root_mean_squared_error, mean_absolute_error


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
    assert not np.isnan(s)


def test_compound_skill(modelresult_oresund_2d, klagshamn, drogden):
    mr = modelresult_oresund_2d

    mr.add_observation(klagshamn, item=0)
    mr.add_observation(drogden, item=0)
    collection = mr.extract()

    assert collection.compound_skill(metric=root_mean_squared_error) > 0.0
    report = collection.skill_df(metrics=[root_mean_squared_error, mean_absolute_error])


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


def test_misc_properties(klagshamn, drogden):

    mr = ModelResult("tests/testdata/Oresund2D.dfsu")

    mr.add_observation(klagshamn, item=0)
    mr.add_observation(drogden, item=0)

    c = mr.extract()

    assert len(c) == 2
    assert c.n_comparisons == 2

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


def test_skill_df(klagshamn, drogden):

    mr = ModelResult("tests/testdata/Oresund2D.dfsu")

    mr.add_observation(klagshamn, item=0)
    mr.add_observation(drogden, item=0)

    c = mr.extract()

    df = c.skill_df()
    assert df.loc["Klagshamn"].n == 167

    # Filtered skill_df
    df = c.skill_df(observation="Klagshamn")
    assert df.loc["Klagshamn"].n == 167
