import pytest
import polars as pl
import pandas as pd
import matplotlib as mpl

import modelskill as ms

# use non-interactive backend for testing
mpl.use("Agg")


@pytest.fixture
def sk_df1() -> pl.DataFrame:
    d = {
        "observation": "obs1",
        "model": "m1",
        "n": 123,
        "bias": 0.1,
        "rmse": 0.2,
        "corr": 0.3,
        "si": 0.4,
        "r2": 0.5,
    }
    # df = pd.DataFrame(d, index=["obs1"])
    df = pl.DataFrame(d)
    # df.index.name = "observation"
    # df = df.reset_index().set_index(["observation", "model"])
    return df


@pytest.fixture
def sk_df2():
    d = {
        "n": [123, 456],
        "x": [1.1, 2.1],
        "y": [1.2, 2.2],
        "bias": [1.3, 2.3],
        "rmse": [1.4, 2.4],
        "corr": [1.5, 2.5],
    }
    df = pd.DataFrame(d, index=["obs1", "obs2"])
    df.index.name = "observation"
    return df


@pytest.fixture
def cmp1() -> ms.Comparer:
    fn = "tests/testdata/NorthSeaHD_and_windspeed.dfsu"
    mr = ms.model_result(fn, item=0, name="HD")
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    df = pd.read_csv(fn, index_col=0, parse_dates=True)
    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        o1 = ms.TrackObservation(df, item=2, name="alti")
    return ms.match(o1, mr)


@pytest.fixture
def o1():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return ms.PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def o2():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return ms.PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def o3():
    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    return ms.TrackObservation(fn, item=3, name="c2")


@pytest.fixture
def cc2(o1, o2, o3):
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    mr1 = ms.model_result(fn, item=0, name="SW_1")
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    mr2 = ms.model_result(fn, item=0, name="SW_2")
    return ms.match([o1, o2, o3], [mr1, mr2], spatial_method="nearest")


def test_skill_table(sk_df1):
    sk = ms.SkillTable(sk_df1)
    assert sk.obs_names == ["obs1"]
    assert sk.mod_names == ["m1"]
    assert sk.quantity_names == []
    # TODO not sure this is assertion is meaningful
    # assert sk.metrics == ["n", "bias", "rmse", "corr", "si", "r2"]
    # This should be sufficient
    metrics = ["n", "bias", "rmse", "corr", "si", "r2"]
    assert all([m in sk.metrics for m in metrics])


def test_skill_repr_html(sk_df1):
    sk = ms.SkillTable(sk_df1)
    repr_html = sk._repr_html_()
    assert "obs1" in repr_html


def test_skill_table_odd_index(sk_df2):
    # having a different index name works
    df = pl.from_pandas(sk_df2.reset_index())
    sk = ms.SkillTable(df)
    assert set(sk.obs_names) == set(["obs1", "obs2"])
    # TODO revisit this, I think there should one model
    assert sk.mod_names == []
    # TODO revisit this, I think there should one quantity
    assert sk.quantity_names == []
    assert set(sk.metrics) == set(["n", "bias", "rmse", "corr"])  # note: no "x", "y"


def test_skill_table_from_xarray(sk_df2):
    ds = sk_df2.to_xarray()
    sk = ms.SkillTable(ds)
    # TODO what is the value of .obs_names?
    assert "obs1" in sk.obs_names
    assert "obs2" in sk.obs_names
    assert sk.mod_names == []
    assert sk.quantity_names == []
    assert set(sk.metrics) == set(["n", "bias", "rmse", "corr"])  # note: no "x", "y"


def test_skill(cmp1):
    sk = cmp1.skill()

    # TODO a minimal skill assesment consists of 1 observation, 1 model and 1 variable
    # in this case model and variable is implict since we only have one of each, but why do we have one observation, seems inconsistent

    assert len(sk.mod_names) == 0  # TODO seems wrong
    assert len(sk.obs_names) == 1  # makes sense
    assert len(sk.quantity_names) == 0  # TODO seems wrong

    df = sk.to_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert "bias" in repr(sk)


def test_skill_bad_args(cmp1: ms.Comparer):
    with pytest.raises(TypeError):
        cmp1.skill(nonexisting_arg=1)  # type: ignore


def test_skill_multi_model(cc2):
    sk = cc2.skill(metrics=["rmse", "bias"])

    # TODO decide if N is a metric or not🤔
    # TODO not a meaningful assertion
    # assert len(sk.metrics) == 3

    assert len(sk.mod_names) == 2
    assert len(sk.obs_names) == 3

    # TODO recreate functionality of xs more domain specific
    # s2 = s.xs("SW_1", level="model")
    # assert len(s2.mod_names) == 0

    # s2 = s.xs("c2", level="observation")
    # assert len(s2.obs_names) == 0

    # s2 = s.reorder_levels(["observation", "model"])
    # assert np.all(s2.index.levels[0] == s.index.levels[1])


# def test_skill_mm_swaplevel(cc2):
#     sk = cc2.skill(metrics=["rmse", "bias"])
#     assert list(sk.data.index.names) == ["model", "observation"]
#     sk2 = sk.swaplevel()
#     assert np.all(sk2.index.levels[0] == sk.index.levels[1])


# @pytest.mark.skipif(pd.__version__ < "2.0.0", reason="requires newer pandas")
# def test_skill_mm_sort_index(cc2):
#     sk = cc2.skill(metrics=["rmse", "bias"])
#     assert list(sk.index.get_level_values(1)) == [
#         "HKNA",
#         "EPL",
#         "c2",
#         "HKNA",
#         "EPL",
#         "c2",
#     ]

#     sk2 = sk.sort_index(level="observation")
#     assert list(sk2.index.get_level_values(1)) == [
#         "EPL",
#         "EPL",
#         "HKNA",
#         "HKNA",
#         "c2",
#         "c2",
#     ]

#     sk3 = sk.swaplevel().sort_index()
#     assert list(sk3.index.get_level_values(0)) == [
#         "EPL",
#         "EPL",
#         "HKNA",
#         "HKNA",
#         "c2",
#         "c2",
#     ]


def test_skill_mm_sort(cc2):
    sk = cc2.skill(metrics=["rmse", "bias"])

    sk2 = sk.sort("rmse")
    # best rmse is for SW_1, EPL
    assert sk2[0, "model"] == "SW_1"
    assert sk2[0, "observation"] == "EPL"
    # worst rmse is for SW_2, c2
    assert sk2[-1, "model"] == "SW_2"
    assert sk2[-1, "observation"] == "c2"

    sk3 = sk.sort("rmse", descending=True)
    assert sk3[0, "model"] == "SW_2"
    assert sk3[0, "observation"] == "c2"
    assert sk3[-1, "model"] == "SW_1"
    assert sk3[-1, "observation"] == "EPL"

    sk4 = sk.sort(["n", "rmse"])
    assert sk4[0, "model"] == "SW_1"
    assert sk4[0, "observation"] == "EPL"
    assert sk4[-1, "model"] == "SW_1"
    assert sk4[-1, "observation"] == "HKNA"


def test_skill_sel(cmp1):
    sk = cmp1.skill(metrics=["rmse", "bias"])
    s2 = sk.sel(observation="alti")
    assert len(s2) == 1
    assert "rmse" in sk.metrics
    assert "bias" in sk.metrics


def test_skill_sel_metrics_str(cmp1):
    sk = cmp1.skill(metrics=["rmse", "bias"])

    assert "rmse" in sk.metrics


def test_skill_sel_multi_model(cc2):
    sk = cc2.skill(metrics=["rmse", "bias"])
    sk2 = sk.sel(model="SW_1")
    assert len(sk2.mod_names) == 1

    # Below doesn't make sense, there is always at least one observation, one model and one quantity
    # assert len(sk2.mod_names) == 0  # no longer in index
    # assert not isinstance(s2.index, pd.MultiIndex)
    assert len(sk2) == 3

    sk2 = sk.sel(model="SW_1", observation=["EPL", "c2"])
    assert len(sk2.obs_names) == 2
    # assert not isinstance(s2.index, pd.MultiIndex)
    assert len(sk2) == 2

    sk2 = sk.sel(model=1, observation=["EPL"])
    assert len(sk2.obs_names) == 1
    # assert not isinstance(s2.index, pd.MultiIndex)
    assert len(sk2) == 1


# def test_skill_sel_query(cc2):
#     sk = cc2.skill(metrics=["rmse", "bias"])
#     with pytest.warns(FutureWarning, match="deprecated"):
#         sk2 = sk.sel(query="rmse>0.2")

#     assert len(sk2.mod_names) == 2

#     # s2 = s.sel("rmse>0.2", model="SW_2", observation=[0, 2])
#     # assert len(s2.mod_names) == 0  # no longer in index


def test_skill_sel_fail(cc2):
    sk = cc2.skill(metrics=["rmse", "bias"])

    with pytest.raises(KeyError):
        sk.sel(quantity="Hm0")

    with pytest.raises(IndexError):
        sk.sel(model=99)


def test_skill_plot_bar(cmp1):
    sk = cmp1.skill(metrics=["rmse", "bias"])
    sk["bias"].plot.bar()


def test_skill_plot_bar_multi_model(cc2):
    sk = cc2.skill(metrics="rmse")
    sk["rmse"].plot.bar()

    with pytest.raises(KeyError):
        sk["bad_metric"].plot.bar()


def test_skill_plot_line(cmp1):
    sk = cmp1.skill(metrics=["rmse", "bias"])
    sk["bias"].plot.line()
    sk["bias"].plot.line(title="Skill")

    with pytest.raises(KeyError):
        sk["NOT_A_METRIC"].plot.line()


def test_skill_plot_line_multi_model(cc2):
    sk = cc2.skill(metrics="rmse")
    sk.rmse.plot.line()

    with pytest.raises(KeyError):
        sk["bad_metric"]


def test_skill_plot_grid(cc2):
    sk = cc2.skill()
    sk["rmse"].plot.grid()
    sk["bias"].plot.grid()
    sk["si"].plot.grid(fmt=".0%")
    sk["bias"].plot.grid(figsize=(2, 1), show_numbers=False)

    sk2 = sk.sel(model="SW_1")
    with pytest.warns(UserWarning) as wn:
        sk2["rmse"].plot.grid()
    assert len(wn) == 1
    assert "only possible for MultiIndex" in str(wn[0].message)


@pytest.mark.skipif(pd.__version__ < "1.5.0", reason="requires Pandas 1.5.0 or higher")
def test_skill_style(cc2):
    sk = cc2.skill(metrics=["bias", "rmse", "lin_slope", "si"])
    sk.style()
    sk.style(decimals=0)
    sk.style(metrics="rmse")
    sk.style(metrics=["bias", "rmse"])
    sk.style(metrics=[])
    sk.style(cmap="viridis_r", show_best=False)


@pytest.mark.skipif(pd.__version__ < "1.5.0", reason="requires Pandas 1.5.0 or higher")
def test_styled_skill_can_be_rendered(cc2):
    sk = cc2.skill()
    # _repr_html_ is called by Jupyter
    sk._repr_html_()

    styled_skill = cc2.skill().style()
    styled_skill._repr_html_()


def test_skill_round(cc2):
    sk = cc2.skill()

    # TODO consider decimals per metric, e.g. {bias: 2, rmse: 3}
    sk.round(decimals=2)
