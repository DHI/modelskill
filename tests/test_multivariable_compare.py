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
    cc1 = ms.match([o1, o2, o3], mr1Hm0)
    cc2 = ms.match([wind1, wind2, wind3], mr1WS)
    return cc1 + cc2


@pytest.fixture
def cc(mr1Hm0, mr1WS, mr2Hm0, mr2WS, o1, o2, o3, wind1, wind2, wind3):
    cc1 = ms.match([o1, o2, o3], [mr1Hm0, mr2Hm0])
    cc2 = ms.match([wind1, wind2, wind3], [mr1WS, mr2WS])
    return cc1 + cc2


def test_n_quantities(cc):
    assert cc.n_quantities == 2


def test_mv_skill(cc_1model: ms.ComparerCollection) -> None:
    df = cc_1model.skill().to_dataframe()
    assert "observation" in df.columns
    assert "quantity" in df.columns
    assert df.filter(observation="EPL_Hm0", quantity="Significant wave height")[
        0, "rmse"
    ] == pytest.approx(0.22492342229)
    assert df.filter(observation="HKNA_wind", quantity="Wind speed")[
        0, "rmse"
    ] == pytest.approx(1.305358970478)
    # spatial_interp nearest: 0.22359663 and 1.2761789


def test_mv_mm_skill(cc):
    df = cc.skill().to_dataframe()
    assert "model" in df.columns
    assert "observation" in df.columns
    assert "quantity" in df.columns
    # spatial_interp nearest: 1.27617894455
    df.filter(model="SW_1", observation="HKNA_wind", quantity="Wind speed")["rmse"][
        0
    ] == pytest.approx(1.30535897)

    df = cc.sel(model="SW_1").skill().to_dataframe()
    assert "observation" in df.columns
    assert "quantity" in df.columns
    assert df.filter(observation="EPL_Hm0")[0, "rmse"] == pytest.approx(0.2249234222997)
    # spatial interp nearest: 0.22359663
    assert df.filter(observation="HKNA_wind", quantity="Wind speed")[
        0, "rmse"
    ] == pytest.approx(1.30535897)

    df = cc.sel(quantity="Wind speed").skill().to_dataframe()
    assert "model" in df.columns
    assert "observation" in df.columns
    assert df.filter(model="SW_1", observation="HKNA_wind")["rmse"][0] == pytest.approx(
        1.30535897
    )


def test_mv_mm_mean_skill(cc):
    df = cc.mean_skill().to_dataframe()
    # idx = ("SW_1", "Wind speed")
    # assert pytest.approx(df.loc[idx].r2) == 0.643293404624
    assert df.filter(model="SW_1", quantity="Wind speed")[0, "r2"] == pytest.approx(
        0.643293404624
    )
    # spatial_interp nearest: 0.63344531

    df = cc.sel(quantity="Significant wave height").mean_skill().to_dataframe()
    assert df.filter(model="SW_1")[0, "cc"] == pytest.approx(0.9640104274957)
    # spatial_interp nearest: 0.963095


def test_mv_mm_scatter(cc):
    cc.sel(model="SW_1", quantity="Wind speed").plot.scatter()
    cc.sel(model="SW_1", quantity="Wind speed").plot.scatter(show_density=True)
    cc.sel(model="SW_1", quantity="Wind speed", observation="F16_wind").plot.scatter(
        skill_table=True
    )
    cc.sel(model="SW_1", quantity="Wind speed").plot.scatter(show_density=True, bins=19)
    cc.sel(model="SW_1", quantity="Wind speed").plot.scatter(show_density=True, bins=21)
    assert True
    plt.close("all")


def cm_1(obs, model):
    """Custom metric #1"""
    return np.mean(obs / model)


def cm_2(obs, model):
    """Custom metric #2"""
    return np.mean(obs * 1.5 / model)

    mtr.add_metric(cm_1)
    mtr.add_metric(cm_2, has_units=True)
    ccs = cc.sel(
        model="SW_1",
        quantity="Wind speed",
        observation="F16_wind",
    )
    ccs.plot.scatter(skill_table=["bias", cm_1, "si", cm_2])
    assert True
    plt.close("all")


def test_mv_mm_taylor(cc):
    cc.sel(quantity="Wind speed").plot.taylor()
    cc.plot.taylor(figsize=(4, 4))
    cc.sel(model="SW_2", start="2017-10-28").plot.taylor()
    assert True
    plt.close("all")
