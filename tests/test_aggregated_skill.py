import pytest
import pandas as pd
import matplotlib as mpl

import modelskill as ms

# use non-interactive backend for testing
mpl.use("Agg")


@pytest.fixture
def cc1():
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
    return ms.match([o1, o2, o3], [mr1, mr2])


def test_skill(cc1):
    s = cc1.skill()
    assert len(s.mod_names) == 0
    assert len(s.obs_names) == 1
    assert len(s.var_names) == 0

    df = s.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "bias" in repr(s)


def test_skill_bad_args(cc1):
    with pytest.raises(AssertionError):
        cc1.skill(nonexisting_arg=1)


def test_skill_multi_model(cc2):
    s = cc2.skill(metrics=["rmse", "bias"])

    # TODO decide if N is a metric or not🤔
    assert len(s.metrics) == 3

    assert len(s.mod_names) == 2
    assert len(s.obs_names) == 3

    # TODO recreate functionality of xs more domain specific
    # s2 = s.xs("SW_1", level="model")
    # assert len(s2.mod_names) == 0

    # s2 = s.xs("c2", level="observation")
    # assert len(s2.obs_names) == 0

    # s2 = s.swaplevel()
    # assert np.all(s2.index.levels[0] == s.index.levels[1])

    # s2 = s.head(1)
    # assert s.iloc[0]["rmse"] == s2.iloc[-1]["rmse"]

    # s2 = s.tail(1)
    # assert s.iloc[-1]["rmse"] == s2.iloc[0]["rmse"]

    # s2 = s.sort_index(level="observation")
    # assert np.all(s2.iloc[0].name == ("SW_1", "EPL"))

    # s2 = s.reorder_levels(["observation", "model"])
    # assert np.all(s2.index.levels[0] == s.index.levels[1])


def test_skill_sel(cc1):
    s = cc1.skill(metrics=["rmse", "bias"])
    s2 = s.sel(observation="alti")
    assert len(s2) == 1
    assert "rmse" in s.metrics
    assert "bias" in s.metrics


def test_skill_sel_metrics_str(cc1):
    s = cc1.skill(metrics=["rmse", "bias"])

    with pytest.warns(FutureWarning, match="deprecated"):
        s2 = s.sel(metrics="rmse")
    assert s2.data.name == "rmse"


def test_skill_sel_metrics_list(cc2):
    s = cc2.skill(metrics=["rmse", "bias"])

    with pytest.warns(FutureWarning, match="deprecated"):
        s2 = s.sel(metrics=["rmse", "n"])
    assert "n" in s2.metrics
    assert "bias" not in s2.metrics


def test_skill_sel_multi_model(cc2):
    s = cc2.skill(metrics=["rmse", "bias"])
    s2 = s.sel(model="SW_1")
    assert len(s2.mod_names) == 0  # no longer in index
    # assert not isinstance(s2.index, pd.MultiIndex)
    assert len(s2) == 3

    s2 = s.sel(model="SW_1", observation=["EPL", "c2"])
    assert len(s2.obs_names) == 2
    # assert not isinstance(s2.index, pd.MultiIndex)
    assert len(s2) == 2

    s2 = s.sel(model=1, observation=["EPL"])
    assert len(s2.obs_names) == 0
    # assert not isinstance(s2.index, pd.MultiIndex)
    assert len(s2) == 1


def test_skill_sel_query(cc2):
    s = cc2.skill(metrics=["rmse", "bias"])
    with pytest.warns(FutureWarning, match="deprecated"):
        s2 = s.sel(query="rmse>0.2")
    
    assert len(s2.mod_names) == 2

    # s2 = s.sel("rmse>0.2", model="SW_2", observation=[0, 2])
    # assert len(s2.mod_names) == 0  # no longer in index


def test_skill_sel_fail(cc2):
    s = cc2.skill(metrics=["rmse", "bias"])
    with pytest.raises(KeyError):
        s.sel(metrics=["cc"])

    with pytest.raises(KeyError):
        s.sel(variable="Hm0")

    with pytest.raises(KeyError):
        s.sel(model=99)


def test_skill_plot_bar(cc1):
    s = cc1.skill(metrics=["rmse", "bias"])
    s["bias"].plot.bar()


def test_skill_plot_bar_multi_model(cc2):
    s = cc2.skill(metrics="rmse")
    s["rmse"].plot.bar()

    with pytest.raises(KeyError):
        s["bad_metric"].plot.bar()


def test_skill_plot_line(cc1):
    s = cc1.skill(metrics=["rmse", "bias"])
    s["bias"].plot.line()
    s["bias"].plot.line(title="Skill")

    with pytest.raises(KeyError):
        s["NOT_A_METRIC"].plot.line()


def test_skill_plot_line_multi_model(cc2):
    s = cc2.skill(metrics="rmse")
    s.rmse.plot.line()

    with pytest.raises(KeyError):
        s["bad_metric"]


def test_skill_plot_grid(cc2):
    s = cc2.skill()
    s["rmse"].plot.grid()
    s["bias"].plot.grid()
    s["si"].plot.grid(fmt=".0%")
    s["bias"].plot.grid(figsize=(2, 1), show_numbers=False)

    s2 = s.sel(model="SW_1")
    with pytest.warns(UserWarning) as wn:
        s2["rmse"].plot.grid()
    assert len(wn) == 1
    assert "only possible for MultiIndex" in str(wn[0].message)


def test_skill_style(cc2):
    s = cc2.skill(metrics=["bias", "rmse", "lin_slope", "si"])
    s.style()
    s.style(decimals=0)
    s.style(metrics="rmse")
    s.style(metrics=["bias", "rmse"])
    s.style(metrics=[])
    s.style(cmap="viridis_r", show_best=False)


def test_skill_round(cc2):
    s = cc2.skill()

    # TODO consider decimals per metric, e.g. {bias: 2, rmse: 3}
    s.round(decimals=2)
