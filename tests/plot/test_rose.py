import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import numpy as np
import pytest

import mikeio
from modelskill.rose import wind_rose, pretty_intervals, directional_labels


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


def test_rose(wave_data_model_obs):
    data = wave_data_model_obs.to_numpy()
    ax = wind_rose(data, mag_step=0.25, cmap1="jet")
    assert ax is not None


def test_pretty_intervals_respects_defined_intervals():
    data1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    data2 = np.array([0.01, 0.2, 0.3, 0.4, 0.5, 0.6])
    ui, vmin, vmax = pretty_intervals(
        data_1=data1, data_2=data2, mag_bins=[0.1, 0.2, 0.45]
    )
    assert vmin == 0.1
    assert vmax == 0.95  # TODO why?

    # The computed intervals has one more than the mag_bins
    assert list(ui[:-1]) == [0.1, 0.2, 0.45]

    # The last interval is larger than the second last
    assert ui[-1] > ui[-2]


def test_pretty_intervals():
    data1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    data2 = np.array([0.01, 0.2, 0.3, 0.4, 0.5, 0.6])
    ui, vmin, vmax = pretty_intervals(
        data_1=data1, data_2=data2, vmin=0.2, mag_step=0.1
    )
    # TODO WIP
    assert vmin == 0.2
    assert vmax == 0.5  # TODO is this correct?
    assert len(ui) == 3

def test_directional_labels():

    assert directional_labels(4) == ('N', 'E', 'S', 'W')
    assert directional_labels(8) == ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')
    assert directional_labels(16) == ('N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW')

    with pytest.raises(ValueError):
        directional_labels(5)

def test_wind_rose_image_identical(wave_data_model_obs, tmp_path):
    # TODO this test seems fragile, since it relies pixel by pixel comparison of images
    data = wave_data_model_obs.to_numpy()
    wind_rose(data)
    
    baseline_path = "tests/baseline/wind_rose_defaults.png"
    img_path = tmp_path / "temp.png"
    fig = plt.gcf()
    fig.set_size_inches(10, 6) # TODO without setting the size, the legends are outside the image
    plt.tight_layout()
    # plt.savefig(baseline_path) # uncomment to generate new baseline
    plt.savefig(img_path)
    
    # compare images to ensure that the plot is identical to the baseline pixel by pixel
    
    baseline_arr = np.array(Image.open(baseline_path))
    img_arr = np.array(Image.open(img_path))

    # these two Numpy arrays should be the same
    assert np.all(baseline_arr == img_arr)

    



    
    

