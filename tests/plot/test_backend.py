import sys

import plotly.graph_objects as go
import pytest

from modelskill.plotting._backend import (
    BACKENDS,
    apply_layout,
    directional_ticks,
    figsize_to_layout,
    import_plotly_go,
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


def test_import_plotly_go_missing_dependency_gives_actionable_error(monkeypatch):
    # setting a sys.modules entry to None makes the import raise ImportError
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)

    with pytest.raises(ImportError, match=r'pip install "modelskill\[plotly\]"'):
        import_plotly_go()


def test_figsize_is_translated_to_plotly_pixels():
    assert figsize_to_layout(None) == {}
    assert figsize_to_layout((8, 6)) == {"width": 800, "height": 600}


def test_apply_layout_uses_figsize_for_width_and_height():
    fig = apply_layout(go.Figure(), figsize=(3, 4))

    assert fig.layout.width == 300
    assert fig.layout.height == 400


def test_apply_layout_lets_explicit_width_win_over_figsize():
    fig = apply_layout(go.Figure(), figsize=(3, 4), width=1000)

    assert fig.layout.width == 1000
    assert fig.layout.height == 400


def test_apply_layout_ignores_none_values():
    fig = apply_layout(go.Figure(), figsize=None, title=None)

    assert fig.layout.width is None
    assert fig.layout.title.text is None


def test_apply_layout_names_the_offending_matplotlib_argument():
    with pytest.raises(ValueError, match="Invalid plotly layout argument: 'cmap'"):
        apply_layout(go.Figure(), cmap="OrRd")


def test_reject_matplotlib_axes_only_for_other_backends():
    reject_matplotlib_axes(None, "plotly")
    reject_matplotlib_axes("some axes", "matplotlib")

    with pytest.raises(ValueError, match="Cannot pass matplotlib axes"):
        reject_matplotlib_axes("some axes", "plotly")


def test_directional_ticks_cover_the_compass():
    assert list(directional_ticks()) == [0, 45, 90, 135, 180, 225, 270, 315, 360]
    assert list(directional_ticks(lim=(90, 180))) == [90, 135, 180]
