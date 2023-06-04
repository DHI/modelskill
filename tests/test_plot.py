import numpy as np
import pytest

from modelskill.plot import sample_points


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
