import pytest
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

import modelskill as ms

# from modelskill.metrics import root_mean_squared_error, mean_absolute_error
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
    repr_text = repr(ccs)
    assert "<ComparerCollection>" in repr_text
    assert "Klagshamn" in repr_text
    assert "dmi_30357_Drogden_Fyr" in repr_text

    ccs2 = cc[["dmi_30357_Drogden_Fyr", "Klagshamn"]]
    repr_text = repr(ccs2)
    assert len(ccs2)
    assert "<ComparerCollection>" in repr_text
    assert "Klagshamn" in repr_text
    assert "dmi_30357_Drogden_Fyr" in repr_text


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
    # assert not np.any(np.isnan(df))
    assert df.null_count().to_numpy().sum() == 0


def test_score_two_elements() -> None:
    mr = ms.model_result("tests/testdata/two_elements.dfsu", item=0)

    obs_df = pd.DataFrame(
        [2.0, 2.0], index=pd.date_range("2020-01-01", periods=2, freq="D")
    )

    # observation is in the center of the second element
    obs = ms.PointObservation(obs_df, item=0, x=2.0, y=2.0, name="obs")

    cmp: ms.Comparer = ms.match(obs, mr, spatial_method="contained")

    assert cmp.score()["two_elements"] == pytest.approx(0.0)

    cmp_default: ms.Comparer = ms.match(obs, mr)

    assert cmp_default.score()["two_elements"] == pytest.approx(0.0)


def test_score(modelresult_oresund_WL, klagshamn, drogden) -> None:
    mr = modelresult_oresund_WL

    cc: ms.ComparerCollection = ms.match([klagshamn, drogden], mr)

    assert cc.score()["Oresund2D_subset"] == pytest.approx(0.198637164895926)
    sk = cc.skill(metrics=["rmse", "mae"])
    # TODO is it appropriate to reach this deep into the data?
    sk.rmse.data.mean()[0, 0] == pytest.approx(0.198637164895926)


def test_weighted_score(modelresult_oresund_WL):
    o1 = ms.PointObservation(
        "tests/testdata/smhi_2095_klagshamn.dfs0",
        item=0,
        x=366844,
        y=6154291,
        name="Klagshamn",
    )
    o2 = ms.PointObservation(
        "tests/testdata/dmi_30357_Drogden_Fyr.dfs0",
        item=0,
        x=355568.0,
        y=6156863.0,
        quantity=ms.Quantity(
            "Water Level", unit="meter"
        ),  # not sure if this is relevant in this test
    )

    mr = ms.model_result("tests/testdata/Oresund2D_subset.dfsu", item=0, name="Oresund")

    cc = ms.match(obs=[o1, o2], mod=mr, spatial_method="contained")
    unweighted = cc.score()
    assert unweighted["Oresund"] == pytest.approx(0.1986296)

    # Weighted

    o1_w = ms.PointObservation(
        "tests/testdata/smhi_2095_klagshamn.dfs0",
        item=0,
        x=366844,
        y=6154291,
        name="Klagshamn",
        weight=10.0,
    )

    o2_w = ms.PointObservation(
        "tests/testdata/dmi_30357_Drogden_Fyr.dfs0",
        item=0,
        x=355568.0,
        y=6156863.0,
        quantity=ms.Quantity(
            "Water Level", unit="meter"
        ),  # not sure if this is relevant in this test
        weight=0.1,
    )

    cc_w = ms.match(obs=[o1_w, o2_w], mod=mr, spatial_method="contained")
    weighted = cc_w.score()

    assert weighted["Oresund"] == pytest.approx(0.1666888)


def test_weighted_score_from_prematched():
    df = pd.DataFrame(
        {"Oresund": [0.0, 1.0], "klagshamn": [0.0, 1.0], "drogden": [-1.0, 2.0]}
    )

    cmp1 = ms.from_matched(
        df[["Oresund", "klagshamn"]],
        mod_items=["Oresund"],
        obs_item="klagshamn",
        weight=100.0,
    )
    cmp2 = ms.from_matched(
        df[["Oresund", "drogden"]],
        mod_items=["Oresund"],
        obs_item="drogden",
        weight=0.0,
    )
    assert cmp1.weight == 100.0
    assert cmp2.weight == 0.0
    assert cmp1.score()["Oresund"] == pytest.approx(0.0)
    assert cmp2.score()["Oresund"] == pytest.approx(1.0)

    cc = ms.ComparerCollection([cmp1, cmp2])
    assert cc["klagshamn"].weight == 100.0
    assert cc["drogden"].weight == 0.0

    assert cc.score()["Oresund"] == pytest.approx(0.0)  # 100 * 0 + 0 * 1


def test_misc_properties(klagshamn, drogden):
    mr = ms.model_result("tests/testdata/Oresund2D_subset.dfsu", item=0)

    cc = ms.match([klagshamn, drogden], mr)

    assert len(cc) == 2

    assert cc.n_models == 1
    assert cc.mod_names == ["Oresund2D_subset"]

    ck = cc["Klagshamn"]
    assert ck.name == "Klagshamn"

    assert ck.n_points > 0

    assert ck.time[0].year == 2018  # intersection of observation and model times
    assert ck.time[-1].year == 2018

    assert ck.x == 366844


# def test_sel_time(cc):
#     c1 = cc["Klagshamn"]
#     c2 = c1.sel(time=slice("2018-01-01", "2018-01-02"))
#     assert c2.start == datetime(2018, 1, 1)


def test_skill(klagshamn, drogden):
    mr = ms.model_result("tests/testdata/Oresund2D_subset.dfsu", item=0)

    cc = ms.match([klagshamn, drogden], mr)

    df = cc.skill().to_dataframe()
    assert df.filter(pl.col("observation") == "Klagshamn")[0, "n"] == 71

    # Filtered skill
    df = cc.sel(observation="Klagshamn").skill().to_dataframe()
    assert df.filter(pl.col("observation") == "Klagshamn")[0, "n"] == 71


def test_skill_choose_metrics(klagshamn, drogden):
    mr = ms.model_result("tests/testdata/Oresund2D_subset.dfsu", item=0)

    cc: ms.ComparerCollection = ms.match([klagshamn, drogden], mr)

    df = cc.skill(metrics=["mae", "si"]).to_dataframe()

    assert "mae" in df.columns
    assert "rmse" not in df.columns

    # Override defaults
    df = cc.skill(metrics=["urmse", "r2"]).to_dataframe()

    assert "r2" in df.columns
    assert "rmse" not in df.columns


def test_skill_choose_metrics_back_defaults(cc: ms.ComparerCollection) -> None:
    df = cc.skill(metrics=["kge", "nse", "max_error"]).to_dataframe()
    assert "kge" in df.columns
    assert "rmse" not in df.columns

    df = cc.mean_skill(metrics=["kge", "nse", "max_error"]).to_dataframe()
    assert "kge" in df.columns
    assert "rmse" not in df.columns

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
    mr = ms.model_result(
        "tests/testdata/Oresund2D_subset.dfsu", item=0, aux_items="U velocity"
    )
    cmp = ms.match(klagshamn, mr, spatial_method="contained")
    assert "U velocity" in cmp.data.data_vars
    assert cmp.data["U velocity"].values[0] == pytest.approx(-0.0360998)
    assert cmp.data["U velocity"].attrs["kind"] == "aux"
    assert cmp.mod_names == ["Oresund2D_subset"]
