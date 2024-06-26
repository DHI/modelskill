import mikeio
import numpy as np
import pytest

from modelskill.plotting._wind_rose import (
    directional_labels,
    pretty_intervals,
    DirectionalHistogram,
)
from modelskill.plotting import wind_rose


@pytest.fixture
def wave_data_model_obs():
    """
    <mikeio.Dataset>
    items:
    0:  China_Model: Sign. Wave Height <Significant wave height> (meter)
    1:  China_Measured: Sign. Wave Height <Significant wave height> (meter)
    2:  China_Model: Mean Wave Direction <Mean Wave Direction> (degree)
    3:  China_Measured: Mean Wave Direction <Mean Wave Direction> (degree)
    """
    ds = mikeio.read("tests/testdata/wave_dir.dfs0")
    df = ds[[0, 2, 1, 3]].to_dataframe()
    return df


def test_directional_histogram():
    # create a small dataset of magnitude and directions
    mag = np.array([0.1, 1.0, 1.0, 1.0, 1.0])
    dirs = np.array([0.0, 0.0, 90.0, 180.0, 270.0])

    X = np.vstack((mag, dirs)).T

    dh = DirectionalHistogram.create_from_data(
        X,
        ui=np.array([0.2, 0.5, 10.0]),
        dir_step=90.0,
    )

    assert dh.calm == pytest.approx(0.2)
    assert dh.density[1, 0] == pytest.approx(0.2)
    assert dh.density[1, 1] == pytest.approx(0.2)
    assert dh.density[1, 2] == pytest.approx(0.2)
    assert dh.density[1, 3] == pytest.approx(0.2)
    assert dh.dir_centers[0] == pytest.approx(90.0)
    assert dh.dir_centers[1] == pytest.approx(180.0)
    assert dh.dir_centers[2] == pytest.approx(270.0)
    assert dh.dir_centers[3] == pytest.approx(360.0)


def test_rose(wave_data_model_obs):
    data = wave_data_model_obs.to_numpy()
    ax = wind_rose(data, mag_step=0.25, cmap1="jet")
    assert ax is not None


def test_single_rose(wave_data_model_obs):
    data = wave_data_model_obs.to_numpy()
    obs_data = data[:, [2, 3]]
    ncol = obs_data.shape[1]
    assert ncol == 2
    ax = wind_rose(obs_data)
    assert ax is not None


def test_pretty_intervals_respects_defined_intervals():
    data1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    data2 = np.array([0.01, 0.2, 0.3, 0.4, 0.5, 0.6])
    xmax = data1.max()
    ymax = data2.max()
    magmax = max(xmax, ymax)
    ui, vmin, vmax = pretty_intervals(magmax, mag_bins=[0.1, 0.2, 0.45])
    assert vmin == 0.1
    assert vmax == 0.95  # TODO why?

    # The computed intervals has one more than the mag_bins
    assert list(ui[:-1]) == [0.1, 0.2, 0.45]

    # The last interval is larger than the second last
    assert ui[-1] > ui[-2]


def test_pretty_intervals():
    data1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    data2 = np.array([0.01, 0.2, 0.3, 0.4, 0.5, 0.6])
    xmax = data1.max()
    ymax = data2.max()
    magmax = max(xmax, ymax)
    ui, vmin, vmax = pretty_intervals(magmax, vmin=0.2, mag_step=0.1)
    # TODO WIP
    assert vmin == 0.2
    assert vmax == 0.5  # TODO is this correct?
    assert len(ui) == 3


def test_pretty_intervals_single_dataset():
    data1 = np.array([0.5, 0.02, 0.3, 0.4, 0.6, 0.5])
    xmax = data1.max()
    ui, vmin, vmax = pretty_intervals(xmax, vmin=0.2, mag_step=0.1)
    assert vmin == 0.2
    assert len(ui) == 3


def test_directional_labels():
    assert directional_labels(4) == ("N", "E", "S", "W")
    assert directional_labels(8) == ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    assert directional_labels(16) == (
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    )

    with pytest.raises(ValueError):
        directional_labels(5)
