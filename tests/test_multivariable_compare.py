import pytest
import numpy as np

from fmskill.model import ModelResult, ModelResultCollection
from fmskill.observation import PointObservation, TrackObservation


@pytest.fixture
def mr1():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    return ModelResult(fn, name="SW_1")


@pytest.fixture
def mr2():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    return ModelResult(fn, name="SW_2")


@pytest.fixture
def o1():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA_Hm0")


@pytest.fixture
def o2():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL_Hm0")


@pytest.fixture
def o3():
    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    return TrackObservation(fn, item=3, name="c2_Hm0")


@pytest.fixture
def wind1():
    fn = "tests/testdata/SW/HKNA_wind.dfs0"
    return PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA_wind")


@pytest.fixture
def wind2():
    fn = "tests/testdata/SW/F16_wind.dfs0"
    return PointObservation(fn, item=0, x=4.01222, y=54.1167, name="F16_wind")


@pytest.fixture
def wind3():
    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    return TrackObservation(fn, item=2, name="c2_wind")


@pytest.fixture
def mrc(mr1, mr2):
    return ModelResultCollection([mr1, mr2])


@pytest.fixture
def cc_1model(mr1, o1, o2, o3, wind1, wind2, wind3):
    mr = mr1
    _add_Hm0_observations(mr, o1, o2, o3)
    _add_wind_observations(mr, wind1, wind2, wind3)
    return mr.extract()


@pytest.fixture
def cc(mr1, mr2, o1, o2, o3, wind1, wind2, wind3):
    mr = ModelResultCollection([mr1, mr2])
    _add_Hm0_observations(mr, o1, o2, o3)
    _add_wind_observations(mr, wind1, wind2, wind3)
    return mr.extract()


def _add_Hm0_observations(mr, o1, o2, o3):
    mr.add_observation(o1, item="Sign. Wave Height")
    mr.add_observation(o2, item="Sign. Wave Height")
    mr.add_observation(o3, item="Sign. Wave Height")


def _add_wind_observations(mr, wind1, wind2, wind3):
    mr.add_observation(wind1, item="Wind speed")
    mr.add_observation(wind2, item="Wind speed")
    mr.add_observation(wind3, item="Wind speed")


def test_n_variables(cc):
    assert cc.n_variables == 2


def test_mv_skill(cc_1model):
    df = cc_1model.skill()
    assert df.index.names[0] == "observation"
    assert df.index.names[1] == "variable"
    assert pytest.approx(df.iloc[0].rmse) == 0.21635651988
    idx = ("HKNA_wind", "Wind_speed")
    assert pytest.approx(df.loc[idx].rmse) == 1.27617894455


def test_mv_mm_skill(cc):
    df = cc.skill()
    assert df.index.names[0] == "model"
    assert df.index.names[1] == "observation"
    assert df.index.names[2] == "variable"
    idx = ("SW_1", "HKNA_wind", "Wind_speed")
    assert pytest.approx(df.loc[idx].rmse) == 1.27617894455

    df = cc.skill(model="SW_1")
    assert df.index.names[0] == "observation"
    assert df.index.names[1] == "variable"
    assert pytest.approx(df.iloc[0].rmse) == 0.21635651988
    idx = ("HKNA_wind", "Wind_speed")
    assert pytest.approx(df.loc[idx].rmse) == 1.27617894455

    df = cc.skill(variable=1)
    assert df.index.names[0] == "model"
    assert df.index.names[1] == "observation"
    idx = ("SW_1", "HKNA_wind")
    assert pytest.approx(df.loc[idx].rmse) == 1.27617894455


def test_mv_mm_mean_skill(cc):
    df = cc.mean_skill()
    assert df.index.names[0] == "model"
    assert df.index.names[1] == "variable"
    idx = ("SW_1", "Wind_speed")
    assert pytest.approx(df.loc[idx].r2) == 0.98130765692

    df = cc.mean_skill(variable=0)
    assert pytest.approx(df.loc["SW_1"].cc) == 0.972628061122
