import pytest
import numpy as np

import modelskill as ms
from modelskill.metrics import root_mean_squared_error, mean_absolute_error
from modelskill.comparison import Comparer


@pytest.fixture
def klagshamn():
    fn = "tests/testdata/smhi_2095_klagshamn.dfs0"
    return ms.PointObservation(fn, item=0, x=366844, y=6154291, name="Klagshamn")


@pytest.fixture
def drogden():
    fn = "tests/testdata/dmi_30357_Drogden_Fyr.dfs0"
    return ms.PointObservation(
        fn,
        item=0,
        x=355568.0,
        y=6156863.0,
        quantity=ms.Quantity("Water Level", unit="meter"),
    )


@pytest.fixture
def modelresult_oresund_WL():
    return ms.ModelResult("tests/testdata/Oresund2D.dfsu", item=0)


@pytest.fixture
def cc(modelresult_oresund_WL, klagshamn, drogden):
    mr = modelresult_oresund_WL
    return ms.compare([klagshamn, drogden], mr)


def test_get_comparer_by_name(cc):
    assert len(cc) == 2
    assert "Klagshamn" in cc.keys()
    assert "dmi_30357_Drogden_Fyr" in cc.keys()
    assert "Atlantis" not in cc.keys()


def test_get_comparer_by_position(cc):
    cc0 = cc[0]
    assert isinstance(cc0, Comparer)
    assert cc0.name == "Klagshamn"

    cc1 = cc[-1]
    assert isinstance(cc1, Comparer)
    assert cc1.name == "dmi_30357_Drogden_Fyr"

    with pytest.raises(NotImplementedError):
        cc[0:2]

    # ccs = cc[0:2]
    # assert len(ccs) == 2
    # assert isinstance(ccs, ComparerCollection)


def test_iterate_over_comparers(cc):
    assert len(cc) == 2
    for c in cc:
        assert isinstance(c, Comparer)


def test_skill_from_observation_with_missing_values(modelresult_oresund_WL):
    o1 = ms.PointObservation(
        "tests/testdata/eq_ts_with_gaps.dfs0",
        item=0,
        x=366844,
        y=6154291,
        name="Klagshamn",
    )
    mr = modelresult_oresund_WL
    cc = ms.compare(o1, mr)
    df = cc["Klagshamn"].skill().df
    assert not np.any(np.isnan(df))


def test_extraction_no_overlap(modelresult_oresund_WL):
    o1 = ms.PointObservation(
        "tests/testdata/smhi_2095_klagshamn_shifted.dfs0",
        x=366844,
        y=6154291,
        name="Klagshamn",
    )
    mr = modelresult_oresund_WL

    con = ms.Connector(o1, mr, validate=False)
    with pytest.raises(ValueError, match="No data"):
        con.extract()


def test_score(modelresult_oresund_WL, klagshamn, drogden):
    mr = modelresult_oresund_WL

    cc = ms.compare([klagshamn, drogden], mr)

    assert cc.score(metric=root_mean_squared_error) > 0.0
    cc.skill(metrics=[root_mean_squared_error, mean_absolute_error])


# def test_weighted_score(modelresult_oresund_WL, klagshamn, drogden):
#     mr = modelresult_oresund_WL

#     cc = ms.compare([klagshamn, drogden], mr)
#     unweighted_skill = cc.score()

#     con = ms.Connector()

#     con.add(klagshamn, mr, weight=0.9, validate=False)
#     con.add(drogden, mr, weight=0.1, validate=False)
#     cc = con.extract()
#     weighted_skill = cc.score()
#     assert unweighted_skill != weighted_skill

#     obs = [klagshamn, drogden]

#     con = ms.Connector(obs, mr, weight=[0.9, 0.1], validate=False)
#     cc = con.extract()
#     weighted_skill2 = cc.score()

#     assert weighted_skill == weighted_skill2


def test_misc_properties(klagshamn, drogden):
    mr = ms.ModelResult("tests/testdata/Oresund2D.dfsu", item=0)

    cc = ms.compare([klagshamn, drogden], mr)

    assert len(cc) == 2
    assert cc.n_comparers == 2

    assert cc.n_models == 1
    assert cc.mod_names == ["Oresund2D"]

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
    mr = ms.ModelResult("tests/testdata/Oresund2D.dfsu", item=0)

    cc = ms.compare([klagshamn, drogden], mr)

    df = cc.skill().df
    assert df.loc["Klagshamn"].n == 167

    # Filtered skill
    df = cc.skill(observation="Klagshamn").df
    assert df.loc["Klagshamn"].n == 167


def test_skill_choose_metrics(klagshamn, drogden):
    mr = ms.ModelResult("tests/testdata/Oresund2D.dfsu", item=0)

    cc = ms.compare([klagshamn, drogden], mr)

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


def test_obs_attrs_carried_over(klagshamn, modelresult_oresund_WL):
    klagshamn.data.attrs["A"] = "B"  # could also have been added in constructor
    cmp = ms.compare(klagshamn, modelresult_oresund_WL)[0]
    assert cmp.data.attrs["A"] == "B"
