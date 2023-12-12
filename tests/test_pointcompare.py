import pytest
import numpy as np
import pandas as pd
import xarray as xr

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
    return ms.model_result("tests/testdata/Oresund2D_subset.dfsu", item=0)


@pytest.fixture
def cc(modelresult_oresund_WL, klagshamn, drogden):
    mr = modelresult_oresund_WL
    return ms.match([klagshamn, drogden], mr)


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

    ccs = cc[0:2]
    assert len(ccs) == 2
    assert "Klagshamn" in ccs


def test_subset_cc_for_named_comparers(cc):
    cmp = cc["Klagshamn"]
    assert cmp.name == "Klagshamn"

    cmp2 = cc[0]
    assert cmp2.name == "Klagshamn"

    ccs = cc[("Klagshamn", "dmi_30357_Drogden_Fyr")]
    assert len(ccs) == 2
    assert (
        repr(ccs)
        == "<ComparerCollection>\nComparer: Klagshamn\nComparer: dmi_30357_Drogden_Fyr"
    )

    ccs2 = cc[["dmi_30357_Drogden_Fyr", "Klagshamn"]]
    assert len(ccs2)
    assert (
        repr(ccs2)
        == "<ComparerCollection>\nComparer: dmi_30357_Drogden_Fyr\nComparer: Klagshamn"
    )


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
    cmp = ms.match(o1, mr)
    df = cmp.skill().to_dataframe()
    assert not np.any(np.isnan(df))


def test_extraction_no_overlap(modelresult_oresund_WL):
    o1 = ms.PointObservation(
        "tests/testdata/smhi_2095_klagshamn_shifted.dfs0",
        x=366844,
        y=6154291,
        name="Klagshamn",
    )
    mr = modelresult_oresund_WL

    with pytest.warns(FutureWarning, match="modelskill.match"):
        con = ms.Connector(o1, mr, validate=False)
    with pytest.raises(ValueError, match="No data"):
        con.extract()


def test_score(modelresult_oresund_WL, klagshamn, drogden):
    mr = modelresult_oresund_WL

    cc = ms.match([klagshamn, drogden], mr)

    assert cc.score(metric=root_mean_squared_error) > 0.0
    cc.skill(metrics=[root_mean_squared_error, mean_absolute_error])


# def test_weighted_score(modelresult_oresund_WL, klagshamn, drogden):
#     mr = modelresult_oresund_WL

#     cc = ms.match([klagshamn, drogden], mr)
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
    mr = ms.model_result("tests/testdata/Oresund2D_subset.dfsu", item=0)

    cc = ms.match([klagshamn, drogden], mr)

    assert len(cc) == 2
    assert cc.n_comparers == 2

    assert cc.n_models == 1
    assert cc.mod_names == ["Oresund2D_subset"]

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
    mr = ms.model_result("tests/testdata/Oresund2D_subset.dfsu", item=0)

    cc = ms.match([klagshamn, drogden], mr)

    df = cc.skill().to_dataframe()
    assert df.loc["Klagshamn"].n == 71

    # Filtered skill
    df = cc.sel(observation="Klagshamn").skill().to_dataframe()
    assert df.loc["Klagshamn"].n == 71


def test_skill_choose_metrics(klagshamn, drogden):
    mr = ms.model_result("tests/testdata/Oresund2D_subset.dfsu", item=0)

    cc = ms.match([klagshamn, drogden], mr)

    cc.metrics = ["mae", "si"]

    df = cc.skill().to_dataframe()

    assert "mae" in df.columns
    assert "rmse" not in df.columns

    # Override defaults
    df = cc.skill(metrics=["urmse", "r2"]).df

    assert "r2" in df.columns
    assert "rmse" not in df.columns


def test_skill_choose_metrics_back_defaults(cc):
    cc.metrics = ["kge", "nse", "max_error"]

    df = cc.skill().to_dataframe()
    assert "kge" in df.columns
    assert "rmse" not in df.columns

    df = cc.mean_skill().to_dataframe()
    assert "kge" in df.columns
    assert "rmse" not in df.columns

    cc.metrics = None  # go back to defaults

    df = cc.mean_skill().to_dataframe()
    assert "kge" not in df.columns
    assert "rmse" in df.columns


def test_obs_attrs_carried_over(klagshamn, modelresult_oresund_WL):
    klagshamn.data.attrs["A"] = "B"  # could also have been added in constructor
    cmp = ms.match(klagshamn, modelresult_oresund_WL)
    assert cmp.data.attrs["A"] == "B"


def test_obs_aux_carried_over(klagshamn, modelresult_oresund_WL):
    klagshamn.data["aux"] = xr.ones_like(klagshamn.data["Klagshamn"])
    klagshamn.data["aux"].attrs["kind"] = "aux"
    cmp = ms.match(klagshamn, modelresult_oresund_WL)
    assert "aux" in cmp.data
    assert cmp.data["aux"].values[0] == 1.0
    assert cmp.data["aux"].attrs["kind"] == "aux"
    assert cmp.mod_names == ["Oresund2D_subset"]


def test_obs_aux_carried_over_nan(klagshamn, modelresult_oresund_WL):
    cmp1 = ms.match(klagshamn, modelresult_oresund_WL)
    assert cmp1.n_points == 71
    assert cmp1.time[0] == pd.Timestamp("2018-03-04 00:00:00")
    assert cmp1.data["Observation"].values[0] == pytest.approx(-0.11)

    # NaN values in aux should not influence the comparison
    klagshamn.data["aux"] = xr.ones_like(klagshamn.data["Klagshamn"])
    klagshamn.data["aux"].attrs["kind"] = "aux"
    klagshamn.data["aux"].loc["2018-03-04 00:00:00"] = np.nan
    cmp2 = ms.match(klagshamn, modelresult_oresund_WL)
    assert cmp2.n_points == 71
    assert cmp2.time[0] == pd.Timestamp("2018-03-04 00:00:00")
    assert cmp2.data["Observation"].values[0] == pytest.approx(-0.11)


def test_mod_aux_carried_over(klagshamn):
    mr = ms.ModelResult(
        "tests/testdata/Oresund2D_subset.dfsu", item=0, aux_items="U velocity"
    )
    cmp = ms.match(klagshamn, mr)
    assert "U velocity" in cmp.data.data_vars
    assert cmp.data["U velocity"].values[0] == pytest.approx(-0.0360998)
    assert cmp.data["U velocity"].attrs["kind"] == "aux"
    assert cmp.mod_names == ["Oresund2D_subset"]
