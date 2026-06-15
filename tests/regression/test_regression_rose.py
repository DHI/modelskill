import sys
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
import pytest

import mikeio
from modelskill.plotting import wind_rose

# Max allowed RMS pixel difference vs. the baseline. The test is a refactoring
# tripwire for the *default* wind rose, not a pixel-exact lock: this tolerance
# absorbs minor rendering drift (antialiasing, font/patch-level matplotlib
# changes) while still catching real layout/data regressions, which measure
# ~30 RMS. Regenerate the baseline (see below) on a major matplotlib bump that
# legitimately changes rendering.
IMAGE_TOLERANCE = 10


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


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_wind_rose_matches_baseline(wave_data_model_obs, tmp_path):
    data = wave_data_model_obs.to_numpy()
    wind_rose(data)

    baseline_path = "tests/regression/baseline/wind_rose_defaults.png"
    img_path = tmp_path / "wind_rose_defaults.png"

    fig = plt.gcf()
    fig.set_size_inches(
        10, 6
    )  # TODO without setting the size, the legends are outside the image
    plt.tight_layout()
    # To regenerate the baseline (e.g. after a major matplotlib bump that
    # legitimately changes rendering), save to baseline_path instead:
    # plt.savefig(baseline_path)
    plt.savefig(img_path)

    # Compare against the baseline within a tolerance. compare_images returns
    # None on success and an explanatory message on failure, writing a
    # *-failed-diff.png next to img_path for inspection.
    result = compare_images(baseline_path, str(img_path), tol=IMAGE_TOLERANCE)
    assert result is None, result
