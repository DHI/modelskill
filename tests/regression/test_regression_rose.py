import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pytest

import mikeio
from modelskill.plotting.plot import wind_rose


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


def test_wind_rose_image_identical(wave_data_model_obs, tmp_path):
    # TODO this test seems fragile, since it relies pixel by pixel comparison of images
    data = wave_data_model_obs.to_numpy()
    wind_rose(data)

    baseline_path = "tests/regression/baseline/wind_rose_defaults.png"
    img_path = tmp_path / "temp.png"

    fig = plt.gcf()
    fig.set_size_inches(
        10, 6
    )  # TODO without setting the size, the legends are outside the image
    plt.tight_layout()
    # plt.savefig(baseline_path) # uncomment to generate new baseline
    plt.savefig(img_path)

    # compare images to ensure that the plot is identical to the baseline pixel by pixel

    baseline_arr = np.array(Image.open(baseline_path))
    img_arr = np.array(Image.open(img_path))

    # these two Numpy arrays should be the same
    assert np.all(baseline_arr == img_arr)
