import numpy as np
import pytest
import pandas as pd

from modelskill.plot import format_skill_df
from modelskill.plot import sample_points


def test_format_skill_df():

    #
    #    	            n	bias	rmse	urmse	mae	cc	si	r2
    # observation
    # smhi_2095_klagshamn	167	1.033099e-09	0.040645	0.040645	0.033226	0.841135	0.376413	0.706335

    df = pd.DataFrame(
        {
            "n": [167],
            "bias": [1.033099e-09],
            "rmse": [0.040645],
            "urmse": [0.040645],
            "mae": [0.033226],
            "cc": [0.841135],
            "si": [0.376413],
            "r2": [0.706335],
        },
        index=["smhi_2095_klagshamn"],
    )

    lines = format_skill_df(df, units="degC")
    assert "N     =  167" in lines[0]
    assert "BIAS  =  0.00 degC" in lines[1]
    assert "RMSE  =  0.04 degC" in lines[2]
    assert "URMSE =  0.04 degC" in lines[3]
    assert "MAE   =  0.03 degC" in lines[4]
    assert "CC    =  0.84 " in lines[5]

    lines_with_short_units = format_skill_df(df, units="meter")

    assert "N     =  167" in lines_with_short_units[0]
    assert "BIAS  =  0.00 m" in lines_with_short_units[1]
    assert "RMSE  =  0.04 m" in lines_with_short_units[2]
    assert "URMSE =  0.04 m" in lines_with_short_units[3]
    assert "MAE   =  0.03 m" in lines_with_short_units[4]
    assert "CC    =  0.84 " in lines_with_short_units[5]


@pytest.fixture
def x_y():
    np.random.seed(42)
    x = np.random.rand(100000)
    y = np.random.rand(100000)
    return x, y


def test_sample_points_bool_selects_all_points(x_y):
    x, y = x_y

    x_sample, y_sample = sample_points(x, y, include=True)
    assert len(x_sample) == len(x)
    assert len(y_sample) == len(y)


def test_sample_points_bool_selects_no_points(x_y):
    x, y = x_y

    x_sample, y_sample = sample_points(x, y, include=False)
    assert len(x_sample) == 0
    assert len(y_sample) == 0


def test_sample_points_int_selects_n_points(x_y):
    x, y = x_y

    x_sample, y_sample = sample_points(x, y, include=10)
    assert len(x_sample) == 10
    assert len(y_sample) == 10


def test_sample_points_float_selects_fraction_points(x_y):
    x, y = x_y

    x_sample, y_sample = sample_points(x, y, include=0.1)
    assert len(x_sample) == 10000
    assert len(y_sample) == 10000


def test_sample_points_float_raises_error(x_y):
    x, y = x_y

    with pytest.raises(ValueError):
        sample_points(x, y, include=1.1)

    with pytest.raises(ValueError):
        sample_points(x, y, include=-0.1)


def test_sample_points_negative_int_raises_error(x_y):
    x, y = x_y

    with pytest.raises(ValueError):
        sample_points(x, y, include=-1)


def test_sample_points_large_int_uses_all_points(x_y):
    x, y = x_y

    x_sample, y_sample = sample_points(x, y, include=1000000)
    assert len(x_sample) == len(x)
    assert len(y_sample) == len(y)
