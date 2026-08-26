"""Plotting backend selection.

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

This module holds the backend vocabulary only; the plotly renderers and the
plotly layout interop live in `_plotly.py`. plotly is an optional dependency,
install it with ``pip install "modelskill[plotly]"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Tuple

import numpy as np
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    import plotly.graph_objects as go

Backend = Literal["matplotlib", "plotly"]

# What a plot returns depends on the backend: axes for matplotlib, a figure for
# plotly. A few plots (taylor) return a matplotlib figure rather than axes.
PlotResult: TypeAlias = "Axes | go.Figure"
FigureResult: TypeAlias = "Figure | go.Figure"

BACKENDS: Tuple[Backend, ...] = ("matplotlib", "plotly")


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


def directional_ticks(
    lim: Tuple[float, float] | None = None, n_sectors: int = 8
) -> np.ndarray:
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
