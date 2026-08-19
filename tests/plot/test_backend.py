import pytest

from modelskill.plotting._backend import (
    BACKENDS,
    directional_ticks,
    reject_matplotlib_axes,
    validate_backend,
)


def test_backends_are_matplotlib_and_plotly():
    assert set(BACKENDS) == {"matplotlib", "plotly"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_validate_backend_accepts_supported_backends(backend):
    assert validate_backend(backend) == backend


@pytest.mark.parametrize("backend", ["mpl", "plotLY", "bokeh", ""])
def test_validate_backend_rejects_unknown_backend(backend):
    with pytest.raises(ValueError, match="Valid options are"):
        validate_backend(backend)


def test_reject_matplotlib_axes_only_for_other_backends():
    reject_matplotlib_axes(None, "plotly")
    reject_matplotlib_axes("some axes", "matplotlib")

    with pytest.raises(ValueError, match="Cannot pass matplotlib axes"):
        reject_matplotlib_axes("some axes", "plotly")


def test_directional_ticks_cover_the_compass():
    assert list(directional_ticks()) == [0, 45, 90, 135, 180, 225, 270, 315, 360]
    assert list(directional_ticks(lim=(90, 180))) == [90, 135, 180]
