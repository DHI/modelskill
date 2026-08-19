"""Plotting backend selection and plotly interop.

modelskill plots can be rendered by either of two backends:

* ``"matplotlib"`` - static, report-friendly figures, returns
  :class:`matplotlib.axes.Axes`
* ``"plotly"`` - interactive figures with zoom/hover, returns
  :class:`plotly.graph_objects.Figure`

Both backends accept the same plot arguments (``title``, ``figsize``,
``xlim``, ``ylim``, ...). Backend-specific extras are passed via
``**kwargs``: to the underlying matplotlib/pandas call for the
matplotlib backend, and to :meth:`plotly.graph_objects.Figure.update_layout`
for the plotly backend.

plotly is an optional dependency, install it with
``pip install "modelskill[plotly]"``.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Sequence, Tuple

import numpy as np

Backend = Literal["matplotlib", "plotly"]

BACKENDS: Tuple[Backend, ...] = ("matplotlib", "plotly")

# plotly sizes are in pixels, matplotlib figsize is in inches
PIXELS_PER_INCH = 100

_PLOTLY_INSTALL_HINT = (
    "The 'plotly' backend requires the optional plotly package. "
    'Install it with `pip install "modelskill[plotly]"`.'
)


def validate_backend(backend: str) -> Backend:
    """Check that a backend name is supported.

    Parameters
    ----------
    backend : str
        name of the plotting backend

    Returns
    -------
    str
        the validated backend name

    Raises
    ------
    ValueError
        if the backend is not one of the supported backends
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"Invalid backend '{backend}'. Valid options are: {list(BACKENDS)}"
        )
    return backend  # type: ignore[return-value]


def import_plotly_go():
    """Import plotly.graph_objects with an actionable error if it is missing.

    Returns
    -------
    module
        the ``plotly.graph_objects`` module

    Raises
    ------
    ImportError
        if plotly is not installed
    """
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError as e:
        raise ImportError(_PLOTLY_INSTALL_HINT) from e
    return go


def figsize_to_layout(figsize: Tuple[float, float] | None) -> Dict[str, float]:
    """Translate a matplotlib figsize (inches) to plotly width/height (pixels).

    Parameters
    ----------
    figsize : (float, float), optional
        width and height in inches, by default None

    Returns
    -------
    dict
        ``{"width": ..., "height": ...}``, empty if figsize is None
    """
    if figsize is None:
        return {}
    width, height = figsize
    return {"width": width * PIXELS_PER_INCH, "height": height * PIXELS_PER_INCH}


def apply_layout(
    fig: Any,
    *,
    figsize: Tuple[float, float] | None = None,
    **kwargs: Any,
) -> Any:
    """Apply modelskill and user layout arguments to a plotly figure.

    ``figsize`` is translated to plotly's ``width``/``height``; an explicit
    ``width``/``height`` in ``kwargs`` wins. Remaining ``kwargs`` are passed
    to ``fig.update_layout``.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        figure to update
    figsize : (float, float), optional
        width and height in inches, by default None
    **kwargs
        keyword arguments for ``plotly.graph_objects.Figure.update_layout``

    Returns
    -------
    plotly.graph_objects.Figure
        the updated figure

    Raises
    ------
    ValueError
        if a keyword argument is not a valid plotly layout property
    """
    layout = {**figsize_to_layout(figsize), **kwargs}
    layout = {k: v for k, v in layout.items() if v is not None}
    try:
        fig.update_layout(**layout)
    except ValueError as e:
        raise ValueError(_layout_error_message(layout, e)) from e
    return fig


def _layout_error_message(layout: Dict[str, Any], error: ValueError) -> str:
    invalid = _invalid_layout_keys(layout)
    named = ", ".join(f"'{k}'" for k in invalid) if invalid else "argument"
    return (
        f"Invalid plotly layout argument: {named}. The plotly backend passes "
        "keyword arguments to plotly.graph_objects.Figure.update_layout, so "
        "matplotlib-only arguments are not accepted. Valid layout properties "
        "are documented at https://plotly.com/python/reference/layout/.\n"
        f"Original plotly error: {error}"
    )


def _invalid_layout_keys(layout: Dict[str, Any]) -> list[str]:
    go = import_plotly_go()
    invalid = []
    for key in layout:
        try:
            go.Layout(**{key: layout[key]})
        except ValueError:
            invalid.append(key)
    return invalid


def reject_matplotlib_axes(ax: Any, backend: str) -> None:
    """Raise if matplotlib axes are passed to a non-matplotlib backend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        the axes argument given by the user
    backend : str
        the selected backend

    Raises
    ------
    ValueError
        if ``ax`` is not None and the backend is not matplotlib
    """
    if ax is not None and backend != "matplotlib":
        raise ValueError(
            f"Cannot pass matplotlib axes to the '{backend}' backend. "
            f"The '{backend}' backend returns a new figure."
        )


def directional_ticks(lim: Tuple[float, float] | None = None, n_sectors: int = 8):
    """Tick values for a directional (0-360 degrees) axis.

    Parameters
    ----------
    lim : (float, float), optional
        axis limits to clip the ticks to, by default None
    n_sectors : int, optional
        number of sectors, by default 8

    Returns
    -------
    np.ndarray
        tick values
    """
    ticks = np.linspace(0, 360, n_sectors + 1)
    if lim is not None:
        ticks = ticks[(ticks >= lim[0]) & (ticks <= lim[1])]
    return ticks


def directional_axis(
    fig: Any, axis: str, lim: Tuple[float, float] | None = None
) -> None:
    """Make a plotly axis directional (0-360 degrees with sector ticks).

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        figure to update
    axis : str
        "x" or "y"
    lim : (float, float), optional
        axis range, by default None which means (0, 360)
    """
    ticks = directional_ticks(lim)
    update = fig.update_xaxes if axis == "x" else fig.update_yaxes
    if len(ticks) > 2:
        update(tickmode="array", tickvals=ticks)
    update(range=lim if lim is not None else (0, 360))


def series_range(series: Sequence[Any]) -> Tuple[float, float]:
    """Combined min/max across a sequence of arrays, ignoring NaN.

    Parameters
    ----------
    series : Sequence
        arrays to take the range over

    Returns
    -------
    (float, float)
        overall minimum and maximum
    """
    values = np.concatenate([np.asarray(s, dtype=float).ravel() for s in series])
    return float(np.nanmin(values)), float(np.nanmax(values))
