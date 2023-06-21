import pytest
import numpy as np
import pandas as pd
from modelskill import (
    ModelResult,
    PointObservation,
    TrackObservation,
    Connector,
)


@pytest.fixture
def cc1():
    fn = "tests/testdata/NorthSeaHD_and_windspeed.dfsu"
    mr = ModelResult(fn, item=0, name="HD")
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    df = pd.read_csv(fn, index_col=0, parse_dates=True)
    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        o1 = TrackObservation(df, item=2, name="alti")
    con = Connector(o1, mr)
    return con.extract()


@pytest.fixture
def o1():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def o2():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def o3():
    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    return TrackObservation(fn, item=3, name="c2")


@pytest.fixture
def cc2(o1, o2, o3):
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    mr1 = ModelResult(fn, item=0, name="SW_1")
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    mr2 = ModelResult(fn, item=0, name="SW_2")
    con = Connector([o1, o2, o3], [mr1, mr2])
    return con.extract()


def test_skill(cc1):
    s = cc1.skill()
    assert isinstance(s.df, pd.DataFrame)
    assert len(s.mod_names) == 0
    assert len(s.obs_names) == 1
    assert len(s.var_names) == 0

    df = s.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "bias" in repr(s)
    assert s.loc["alti"] is not None

    assert np.all(df.index == s.index)
    assert np.all(df.columns == s.columns)
    assert np.all(df.shape == s.shape)
    assert np.all(df.size == s.size)
    assert df.ndim == s.ndim
    assert len(df) == len(s)
    assert df.loc["alti"]["n"] == s.loc["alti"]["n"]
    assert len(df.to_html()) == len(s.to_html())
    # assert len(df.to_markdown()) == len(s.to_markdown())

    s2 = s.sort_values("rmse")
    assert s2.iloc[0]["rmse"] == s["rmse"].max()

    s2 = s.sort_values("rmse", ascending=False)
    assert s2.iloc[0]["rmse"] == s["rmse"].min()


def test_skill_multi_model(cc2):
    s = cc2.skill(metrics=["rmse", "bias"])
    assert isinstance(s.index, pd.MultiIndex)
    assert len(s.mod_names) == 2
    assert len(s.obs_names) == 3
    assert len(s.field_names) == 3

    s2 = s.xs("SW_1", level="model")
    assert len(s2.mod_names) == 0

    s2 = s.xs("c2", level="observation")
    assert len(s2.obs_names) == 0

    s2 = s.swaplevel()
    assert np.all(s2.index.levels[0] == s.index.levels[1])

    s2 = s.head(1)
    assert s.iloc[0]["rmse"] == s2.iloc[-1]["rmse"]

    s2 = s.tail(1)
    assert s.iloc[-1]["rmse"] == s2.iloc[0]["rmse"]

    s2 = s.sort_index(level="observation")
    assert np.all(s2.iloc[0].name == ("SW_1", "EPL"))

    s2 = s.reorder_levels(["observation", "model"])
    assert np.all(s2.index.levels[0] == s.index.levels[1])


def test_skill_sel(cc1):
    s = cc1.skill(metrics=["rmse", "bias"])
    s2 = s.sel(observation="alti")
    assert len(s2) == 1

    s2 = s.sel(columns="rmse")
    assert s2.columns[-1] == "rmse"


def test_skill_sel_multi_model(cc2):
    s = cc2.skill(metrics=["rmse", "bias"])
    s2 = s.sel(model="SW_1")
    assert len(s2.mod_names) == 0  # no longer in index
    assert not isinstance(s2.index, pd.MultiIndex)
    assert len(s2) == 3

    s2 = s.sel(model="SW_1", observation=["EPL", "c2"])
    assert len(s2.obs_names) == 2
    assert not isinstance(s2.index, pd.MultiIndex)
    assert len(s2) == 2

    s2 = s.sel(model=1, observation=["EPL"])
    assert len(s2.obs_names) == 0
    assert not isinstance(s2.index, pd.MultiIndex)
    assert len(s2) == 1


def test_skill_sel_query(cc2):
    s = cc2.skill(metrics=["rmse", "bias"])
    s2 = s.sel("rmse>0.2")
    assert len(s2.mod_names) == 2

    s2 = s.sel("rmse>0.2", model="SW_2", observation=[0, 2])
    assert len(s2.mod_names) == 0  # no longer in index


def test_skill_sel_columns(cc2):
    s = cc2.skill(metrics=["rmse", "bias"])
    s2 = s.sel(columns=["rmse", "n"])
    assert s2.columns[-1] == "n"
    assert "bias" not in s2.columns

    s2 = s.sel(columns="rmse")
    assert s2.columns[-1] == "rmse"
    assert "bias" not in s2.columns


def test_skill_sel_fail(cc2):
    s = cc2.skill(metrics=["rmse", "bias"])
    with pytest.raises(KeyError):
        s.sel(columns=["cc"])

    with pytest.raises(KeyError):
        s.sel(variable="Hm0")

    with pytest.raises(KeyError):
        s.sel(model=99)


def test_skill_plot_bar(cc1):
    s = cc1.skill(metrics=["rmse", "bias"])
    s.plot_bar("bias")


def test_skill_plot_bar_multi_model(cc2):
    s = cc2.skill(metrics="rmse")
    s.plot_bar("rmse")

    with pytest.raises(KeyError):
        s.plot_bar("bad_metric")


def test_skill_plot_line(cc1):
    s = cc1.skill(metrics=["rmse", "bias"])
    s.plot_line("bias")


def test_skill_plot_line_multi_model(cc2):
    s = cc2.skill(metrics="rmse")
    s.plot_line("rmse")

    with pytest.raises(KeyError):
        s.plot_line("bad_metric")


def test_skill_plot_grid(cc2):
    s = cc2.skill()
    s.plot_grid("rmse")
    s.plot_grid("bias")
    s.plot_grid("si", fmt=".0%")
    s.plot_grid("bias", figsize=(2, 1), show_numbers=False)

    s2 = s.sel(model="SW_1")
    with pytest.warns(UserWarning) as wn:
        s2.plot_grid("rmse")
    assert len(wn) == 1
    assert "only possible for MultiIndex" in str(wn[0].message)


def test_skill_style(cc2):
    s = cc2.skill(metrics=["bias", "rmse", "lin_slope", "si"])
    s.style()
    s.style(precision=0)
    s.style(columns="rmse")
    s.style(columns=["bias", "rmse"])
    s.style(columns=[])
    s.style(cmap="viridis_r", show_best=False)
