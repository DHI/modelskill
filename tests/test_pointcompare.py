import pandas as pd
import pytest
import numpy as np

import modelskill
from modelskill import ModelResult, PointObservation, Connector, Quantity
from modelskill.metrics import root_mean_squared_error, mean_absolute_error
from modelskill.comparison import Comparer, PointComparer


@pytest.fixture
def klagshamn():
    fn = "tests/testdata/smhi_2095_klagshamn.dfs0"
    return PointObservation(fn, item=0, x=366844, y=6154291, name="Klagshamn")


@pytest.fixture
def drogden():
    fn = "tests/testdata/dmi_30357_Drogden_Fyr.dfs0"
    return PointObservation(
        fn,
        item=0,
        x=355568.0,
        y=6156863.0,
        quantity=Quantity("Water Level", unit="meter"),
    )


@pytest.fixture
def modelresult_oresund_WL():
    return ModelResult("tests/testdata/Oresund2D.dfsu", item=0)


@pytest.fixture
def cc(modelresult_oresund_WL, klagshamn, drogden):
    mr = modelresult_oresund_WL
    con = Connector([klagshamn, drogden], mr, validate=False)
    return con.extract()


def test_get_comparer_by_name(modelresult_oresund_WL, klagshamn, drogden):
    mr = modelresult_oresund_WL

    con = Connector([klagshamn, drogden], mr, validate=False)
    cc = con.extract()

    assert len(cc) == 2
    assert "Klagshamn" in cc.keys()
    assert "dmi_30357_Drogden_Fyr" in cc.keys()
    assert "Atlantis" not in cc.keys()


def test_get_comparer_by_position(cc):
    cc0 = cc[0]
    assert isinstance(cc0, PointComparer)
    assert cc0.name == "Klagshamn"

    cc1 = cc[-1]
    assert isinstance(cc1, PointComparer)
    assert cc1.name == "dmi_30357_Drogden_Fyr"

    with pytest.raises(NotImplementedError):
        cc[0:2]

    # ccs = cc[0:2]
    # assert len(ccs) == 2
    # assert isinstance(ccs, ComparerCollection)


def test_iterate_over_comparers(modelresult_oresund_WL, klagshamn, drogden):
    mr = modelresult_oresund_WL

    con = Connector([klagshamn, drogden], mr, validate=False)
    cc = con.extract()

    assert len(cc) == 2
    for c in cc:
        assert isinstance(c, PointComparer)


def test_skill_from_observation_with_missing_values(modelresult_oresund_WL):
    o1 = PointObservation(
        "tests/testdata/eq_ts_with_gaps.dfs0",
        item=0,
        x=366844,
        y=6154291,
        name="Klagshamn",
    )
    mr = modelresult_oresund_WL
    con = Connector(o1, mr)
    c = con.extract()
    df = c["Klagshamn"].skill().df
    assert not np.any(np.isnan(df))


def test_extraction_no_overlap(modelresult_oresund_WL):
    o1 = PointObservation(
        "tests/testdata/smhi_2095_klagshamn_shifted.dfs0",
        x=366844,
        y=6154291,
        name="Klagshamn",
    )
    mr = modelresult_oresund_WL
    with pytest.warns(UserWarning) as wn:
        con = Connector(o1, mr, validate=False)
    assert len(wn) == 1
    assert "No time overlap" in str(wn[0].message)
    # assert "Could not add observation" in str(wn[1].message)
    assert len(con.observations) == 1
    with pytest.warns(UserWarning, match="No overlapping data"):
        c = con.extract()
    assert c.n_comparers == 0


def test_score(modelresult_oresund_WL, klagshamn, drogden):
    mr = modelresult_oresund_WL

    con = Connector([klagshamn, drogden], mr, validate=False)
    cc = con.extract()

    assert cc.score(metric=root_mean_squared_error) > 0.0
    cc.skill(metrics=[root_mean_squared_error, mean_absolute_error])


def test_weighted_score(modelresult_oresund_WL, klagshamn, drogden):
    mr = modelresult_oresund_WL

    con = Connector([klagshamn, drogden], mr, validate=False)
    cc = con.extract()
    unweighted_skill = cc.score()

    con = Connector()

    con.add(klagshamn, mr, weight=0.9, validate=False)
    con.add(drogden, mr, weight=0.1, validate=False)
    cc = con.extract()
    weighted_skill = cc.score()
    assert unweighted_skill != weighted_skill

    obs = [klagshamn, drogden]

    con = Connector(obs, mr, weight=[0.9, 0.1], validate=False)
    cc = con.extract()
    weighted_skill2 = cc.score()

    assert weighted_skill == weighted_skill2


def test_misc_properties(klagshamn, drogden):

    mr = ModelResult("tests/testdata/Oresund2D.dfsu", item=0)

    con = Connector([klagshamn, drogden], mr, validate=False)
    cc = con.extract()

    assert len(cc) == 2
    assert cc.n_comparers == 2

    assert cc.n_models == 1
    assert cc.mod_names == ["Undefined"]

    ck = cc["Klagshamn"]
    assert ck.name == "Klagshamn"

    assert ck.n_points > 0

    assert ck.start.year == 2018  # intersection of observation and model times
    assert ck.end.year == 2018

    assert ck.x == 366844


# def test_sel_time(cc):
#     c1 = cc["Klagshamn"]
#     c2 = c1.sel(time=slice("2018-01-01", "2018-01-02"))
#     assert c2.start == datetime(2018, 1, 1)


def test_skill(klagshamn, drogden):

    mr = ModelResult("tests/testdata/Oresund2D.dfsu", item=0)

    con = Connector([klagshamn, drogden], mr, validate=False)
    cc = con.extract()

    df = cc.skill().df
    assert df.loc["Klagshamn"].n == 167

    # Filtered skill
    df = cc.skill(observation="Klagshamn").df
    assert df.loc["Klagshamn"].n == 167


def test_skill_choose_metrics(klagshamn, drogden):

    mr = ModelResult("tests/testdata/Oresund2D.dfsu", item=0)

    con = Connector([klagshamn, drogden], mr, validate=False)
    cc = con.extract()

    cc.metrics = ["mae", "si"]

    df = cc.skill().df

    assert "mae" in df.columns
    assert "rmse" not in df.columns

    # Override defaults
    df = cc.skill(metrics=["urmse", "r2"]).df

    assert "r2" in df.columns
    assert "rmse" not in df.columns


def test_skill_choose_metrics_back_defaults(cc):

    cc.metrics = ["kge", "nse", "max_error"]

    df = cc.skill().df
    assert "kge" in df.columns
    assert "rmse" not in df.columns

    df = cc.mean_skill().df
    assert "kge" in df.columns
    assert "rmse" not in df.columns

    cc.metrics = None  # go back to defaults

    df = cc.mean_skill().df
    assert "kge" not in df.columns
    assert "rmse" in df.columns


def test_comparison_from_dict():

    # As an alternative to
    # mr = ModelResult()

    # o1 = PointObservation()
    # con = Connector(o1, mr)
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

    con = modelskill.from_config(configuration, validate_eum=False)
    c = con.extract()
    assert len(c) == 2
    assert c.n_comparers == 2
    assert c.n_models == 1


def test_comparison_from_yml():

    con = modelskill.from_config("tests/testdata/conf.yml", validate_eum=False)
    c = con.extract()

    assert len(c) == 2
    assert c.n_comparers == 2
    assert c.n_models == 1
    assert con.observations["Klagshamn"].quantity.name == "Water Level"


def test_comparer_dataframe_without_time_not_allowed(klagshamn):

    mr = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with pytest.raises(ValueError, match="datetime"):
        Comparer(klagshamn, modeldata=mr)
