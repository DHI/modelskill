import pytest
import numpy as np

import fmskill
from fmskill import ModelResult, PointObservation, Connector
from fmskill.metrics import root_mean_squared_error, mean_absolute_error
from fmskill.comparison import PointComparer


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


def test_get_comparer_by_name(modelresult_oresund_2d, klagshamn, drogden):
    mr = modelresult_oresund_2d

    con = Connector([klagshamn, drogden], mr[0], validate=False)
    cc = con.extract()

    assert len(cc) == 2
    assert "Klagshamn" in cc.keys()
    assert "dmi_30357_Drogden_Fyr" in cc.keys()
    assert "Atlantis" not in cc.keys()


def test_iterate_over_comparers(modelresult_oresund_2d, klagshamn, drogden):
    mr = modelresult_oresund_2d

    con = Connector([klagshamn, drogden], mr[0], validate=False)
    cc = con.extract()

    assert len(cc) == 2
    for c in cc:
        assert isinstance(c, PointComparer)


def test_skill_from_observation_with_missing_values(modelresult_oresund_2d):
    o1 = PointObservation(
        "tests/testdata/eq_ts_with_gaps.dfs0",
        item=0,
        x=366844,
        y=6154291,
        name="Klagshamn",
    )
    mr = modelresult_oresund_2d
    con = Connector(o1, mr[0])
    c = con.extract()
    df = c["Klagshamn"].skill().df
    assert not np.any(np.isnan(df))


def test_extraction_no_overlap(modelresult_oresund_2d):
    o1 = PointObservation(
        "tests/testdata/smhi_2095_klagshamn_shifted.dfs0",
        x=366844,
        y=6154291,
        name="Klagshamn",
    )
    mr = modelresult_oresund_2d
    with pytest.warns(UserWarning) as wn:
        con = Connector(o1, mr[0], validate=False)
    assert len(wn) == 1
    assert "No time overlap" in str(wn[0].message)
    # assert "Could not add observation" in str(wn[1].message)
    assert len(con.observations) == 1
    c = con.extract()
    assert c.n_comparers == 0


def test_score(modelresult_oresund_2d, klagshamn, drogden):
    mr = modelresult_oresund_2d

    con = Connector([klagshamn, drogden], mr[0], validate=False)
    cc = con.extract()

    assert cc.score(metric=root_mean_squared_error) > 0.0
    cc.skill(metrics=[root_mean_squared_error, mean_absolute_error])


def test_weighted_score(modelresult_oresund_2d, klagshamn, drogden):
    mr = modelresult_oresund_2d

    con = Connector([klagshamn, drogden], mr[0], validate=False)
    cc = con.extract()
    unweighted_skill = cc.score()

    con = Connector()
    con.add(klagshamn, mr[0], weight=0.9, validate=False)
    con.add(drogden, mr[0], weight=0.1, validate=False)
    cc = con.extract()
    weighted_skill = cc.score()
    assert unweighted_skill != weighted_skill

    obs = [klagshamn, drogden]
    con = Connector(obs, mr[0], weight=[0.9, 0.1], validate=False)
    cc = con.extract()
    weighted_skill2 = cc.score()

    assert weighted_skill == weighted_skill2


def test_misc_properties(klagshamn, drogden):

    mr = ModelResult("tests/testdata/Oresund2D.dfsu")

    con = Connector([klagshamn, drogden], mr[0], validate=False)
    cc = con.extract()

    assert len(cc) == 2
    assert cc.n_comparers == 2

    assert cc.n_models == 1
    assert cc._mod_names == [
        "Oresund2D"
    ]  # TODO this fails when all tests are run, something is spilling over from another test !!

    ck = cc["Klagshamn"]
    assert ck.name == "Klagshamn"

    assert ck.n_points > 0

    assert ck.start.year == 2018  # intersection of observation and model times
    assert ck.end.year == 2018

    assert ck.x == 366844


def test_skill(klagshamn, drogden):

    mr = ModelResult("tests/testdata/Oresund2D.dfsu")

    con = Connector([klagshamn, drogden], mr[0], validate=False)
    cc = con.extract()

    df = cc.skill().df
    assert df.loc["Klagshamn"].n == 167

    # Filtered skill
    df = cc.skill(observation="Klagshamn").df
    assert df.loc["Klagshamn"].n == 167


def test_comparison_from_dict():

    # As an alternative to
    # mr = ModelResult()

    # o1 = PointObservation()
    # con = Connector(o1, mr[0])
    # c = con.extract()

    configuration = dict(
        modelresults=dict(
            HD=dict(
                filename="tests/testdata/Oresund2D.dfsu",
                item=0,
            ),
        ),
        observations=dict(
            klagshamn=dict(
                filename="tests/testdata/obs_two_items.dfs0",
                item=1,
                x=366844,
                y=6154291,
                name="Klagshamn2",
            ),
            Drogden=dict(
                filename="tests/testdata/dmi_30357_Drogden_Fyr.dfs0",
                item=0,
                x=355568.0,
                y=6156863.0,
            ),
        ),
    )
    con = fmskill.from_config(configuration, validate_eum=False)
    c = con.extract()
    assert len(c) == 2
    assert c.n_comparers == 2
    assert c.n_models == 1


def test_comparison_from_yml():

    con = fmskill.from_config("tests/testdata/conf.yml", validate_eum=False)
    c = con.extract()

    assert len(c) == 2
    assert c.n_comparers == 2
    assert c.n_models == 1
    assert con.observations["Klagshamn"].itemInfo.name == "Water Level"
