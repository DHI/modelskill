import pytest
import matplotlib.pyplot as plt
from matplotlib.table import Table
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
def cc(mr1Hm0, mr1WS, mr2Hm0, mr2WS, o1, o2, o3, wind1, wind2, wind3):
    cc1 = ms.match([o1, o2, o3], [mr1Hm0, mr2Hm0])
    cc2 = ms.match([wind1, wind2, wind3], [mr1WS, mr2WS])
    return cc1 + cc2


@pytest.fixture
def ccs(cc):
    ccs = cc.sel(
        model="SW_1",
        quantity="Wind speed",
        observation="F16_wind",
    )
    return ccs


def cm_1(obs, model):
    """Custom metric #1"""
    return np.mean(obs / model)


def cm_2(obs, model):
    """Custom metric #2"""
    return np.mean(obs * 1.5 / model)


@mtr.metric(display_name="MyBias")
def cm_named(obs, model):
    """Custom metric #2"""
    val = np.mean(obs * 1.5 / model)
    sign = "+" if val > 0 else "-"
    return f"{sign}{val:.3f}"


def test_custom_metric_display_name(ccs):
    mtr.add_metric(cm_1)
    mtr.add_metric(cm_2, has_units=True)
    mtr.add_metric(cm_named)

    s = ccs.plot.scatter(
        skill_table=[
            "bias",
            cm_1,
            "si",
            cm_2,
            cm_named,
        ]
    )
    for child in s.get_children():
        if isinstance(child, Table):
            t = child
            break

    assert t._cells[1, 0]._text._text == "BIAS"
    assert t._cells[2, 0]._text._text == "CM_1"
    assert t._cells[3, 0]._text._text == "SI"
    assert t._cells[4, 0]._text._text == "CM_2"
    assert t._cells[5, 0]._text._text == "MyBias"

    plt.close("all")


def test_custom_metric_result(ccs):
    mtr.add_metric(cm_1)
    mtr.add_metric(cm_2, has_units=True)
    mtr.add_metric(cm_named)

    s = ccs.plot.scatter(
        skill_table=[
            "bias",
            cm_1,
            "si",
            cm_2,
            cm_named,
        ]
    )
    for child in s.get_children():
        if isinstance(child, Table):
            t = child
            break

    assert t._cells[1, 2]._text._text == "2.05 m/s"
    assert t._cells[2, 2]._text._text == "0.86  "
    assert t._cells[3, 2]._text._text == "0.14  "
    assert t._cells[4, 2]._text._text == "1.29 m/s"
    assert t._cells[5, 2]._text._text == "+1.293  "

    plt.close("all")
