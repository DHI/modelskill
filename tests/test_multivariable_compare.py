import pytest
import matplotlib.pyplot as plt
import numpy as np

import modelskill as ms
import modelskill.metrics as mtr


@pytest.fixture
def mr1Hm0():
    fn = "tests/testdata/SW/DutchCoast_2017_subset.dfsu"
    return ms.model_result(fn, item="Sign. Wave Height", name="SW_1")


@pytest.fixture
def mr1WS():
    fn = "tests/testdata/SW/DutchCoast_2017_subset.dfsu"
    return ms.model_result(fn, item="Wind speed", name="SW_1")


@pytest.fixture
def mr2Hm0():
    fn = "tests/testdata/SW/DutchCoast_2017_subset.dfsu"
    return ms.model_result(fn, item="Sign. Wave Height", name="SW_2")


@pytest.fixture
def mr2WS():
    fn = "tests/testdata/SW/DutchCoast_2017_subset.dfsu"
    return ms.model_result(fn, item="Wind speed", name="SW_2")


@pytest.fixture
def o1():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return ms.PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA_Hm0")


@pytest.fixture
def o2():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return ms.PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL_Hm0")


@pytest.fixture
def o3():
    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    return ms.TrackObservation(fn, item=3, name="c2_Hm0")


@pytest.fixture
def wind1():
    fn = "tests/testdata/SW/HKNA_wind.dfs0"
    return ms.PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA_wind")


@pytest.fixture
def wind2():
    fn = "tests/testdata/SW/F16_wind.dfs0"
    return ms.PointObservation(fn, item=0, x=4.01222, y=54.1167, name="F16_wind")


@pytest.fixture
def wind3():
    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    return ms.TrackObservation(fn, item=2, name="c2_wind")


@pytest.fixture
def cc_1model(mr1Hm0, mr1WS, o1, o2, o3, wind1, wind2, wind3):
    cc1 = ms.compare([o1, o2, o3], mr1Hm0)
    cc2 = ms.compare([wind1, wind2, wind3], mr1WS)
    return cc1 + cc2


@pytest.fixture
def cc(mr1Hm0, mr1WS, mr2Hm0, mr2WS, o1, o2, o3, wind1, wind2, wind3):
    cc1 = ms.compare([o1, o2, o3], [mr1Hm0, mr2Hm0])
    cc2 = ms.compare([wind1, wind2, wind3], [mr1WS, mr2WS])
    return cc1 + cc2


def test_n_variables(cc):
    assert cc.n_variables == 2


def test_mv_skill(cc_1model):
    df = cc_1model.skill().to_dataframe()
    assert df.index.names[0] == "observation"
    assert df.index.names[1] == "variable"
    assert pytest.approx(df.iloc[0].rmse) == 0.22359663
    idx = ("HKNA_wind", "Wind speed")
    assert pytest.approx(df.loc[idx].rmse) == 1.27617894455


def test_mv_mm_skill(cc):
    df = cc.skill().to_dataframe()
    assert df.index.names[0] == "model"
    assert df.index.names[1] == "observation"
    assert df.index.names[2] == "variable"
    idx = ("SW_1", "HKNA_wind", "Wind speed")
    assert pytest.approx(df.loc[idx].rmse) == 1.27617894455

    df = cc.sel(model="SW_1").skill().to_dataframe()
    assert df.index.names[0] == "observation"
    assert df.index.names[1] == "variable"
    assert pytest.approx(df.iloc[0].rmse) == 0.22359663
    idx = ("HKNA_wind", "Wind speed")
    assert pytest.approx(df.loc[idx].rmse) == 1.27617894455

    df = cc.sel(variable="Wind speed").skill().to_dataframe()
    assert df.index.names[0] == "model"
    assert df.index.names[1] == "observation"
    idx = ("SW_1", "HKNA_wind")
    assert pytest.approx(df.loc[idx].rmse) == 1.27617894455


def test_mv_mm_mean_skill(cc):
    df = cc.mean_skill().to_dataframe()
    assert df.index.names[0] == "model"
    assert df.index.names[1] == "variable"
    idx = ("SW_1", "Wind speed")
    assert pytest.approx(df.loc[idx].r2) == 0.63344531

    df = cc.sel(variable="Significant wave height").mean_skill().to_dataframe()
    assert pytest.approx(df.loc["SW_1"].cc) == 0.963095


def test_mv_mm_scatter(cc):
    cc.sel(model="SW_1", variable="Wind speed").plot.scatter()
    cc.sel(model="SW_1", variable="Wind speed").plot.scatter(show_density=True)
    cc.sel(model="SW_1", variable="Wind speed", observation="F16_wind").plot.scatter(
        skill_table=True
    )
    cc.sel(model="SW_1", variable="Wind speed").plot.scatter(show_density=True, bins=19)
    cc.sel(model="SW_1", variable="Wind speed").plot.scatter(show_density=True, bins=21)
    assert True
    plt.close("all")


def cm_1(obs, model):
    """Custom metric #1"""
    return np.mean(obs.ravel() / model.ravel())


def cm_2(obs, model):
    """Custom metric #2"""
    return np.mean(obs.ravel() * 1.5 / model.ravel())


def test_custom_metric_skilltable_mv_mm_scatter(cc):
    mtr.add_metric(cm_1)
    mtr.add_metric(cm_2, has_units=True)
    ccs = cc.sel(
        model="SW_1",
        variable="Wind speed",
        observation="F16_wind",
    )
    ccs.plot.scatter(skill_table=["bias", cm_1, "si", cm_2])
    assert True
    plt.close("all")


def test_mv_mm_taylor(cc):
    cc.sel(variable="Wind speed").plot.taylor()
    cc.plot.taylor(figsize=(4, 4))
    cc.sel(model="SW_2", start="2017-10-28").plot.taylor()
    assert True
    plt.close("all")
