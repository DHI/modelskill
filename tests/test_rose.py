import numpy as np
import pytest

from modelskill.plotting._wind_rose import hist2d


def test_hist2d():
    # create a small dataset of magnitude and directions
    mag = np.array([0.1, 1.0, 1.0, 1.0, 1.0])
    dirs = np.array([0.0, 0.0, 90.0, 180.0, 270.0])

    X = np.vstack((mag, dirs)).T

    calm, counts, _ = hist2d(
        X,
        ui=np.array([0.2, 0.5, 10.0]),
        dir_step=90.0,
    )

    assert calm == pytest.approx(0.2)

    assert counts.sum() + calm == pytest.approx(1.0)
