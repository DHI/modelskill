from __future__ import annotations
from dataclasses import dataclass
import warnings
from typing import TYPE_CHECKING, Collection

if TYPE_CHECKING:
    import matplotlib.figure

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from ._taylor_diagram_external import TaylorDiagram


@dataclass
class TaylorPoint:
    name: str
    obs_std: float
    std: float
    cc: float
    marker: str
    marker_size: float


def taylor_diagram(
    obs_std: float,
    points: Collection[TaylorPoint] | TaylorPoint,
    figsize: tuple[float, float] = (7, 7),
    obs_text: str = "Observations",
    normalize_std: bool = False,
    ax: Axes | None = None,
    title: str = "Taylor diagram",
) -> matplotlib.figure.Figure:
    """
    Plot a Taylor diagram using the given observations and points.

    Parameters
    -----------
    obs_std : float
        Standard deviation of the observations.
    points : list of TaylorPoint objects or a single TaylorPoint object
        Points to plot on the Taylor diagram.
    figsize : tuple, optional
        Figure size in inches. Default is (7, 7).
    obs_text : str, optional
        Label for the observations. Default is "Observations".
    normalize_std : bool, optional
        Whether to normalize the standard deviation of the points by the standard deviation of the observations. Default is False.
    title : str, optional
        Title of the plot. Default is "Taylor diagram".

    Returns
    --------
    matplotlib.figure.Figure
            The matplotlib figure object
    """

    if figsize[0] != figsize[1]:
        warnings.warn(
            "It is strongly recommended that the aspect ratio is 1:1 for Taylor diagrams"
        )
    fig = plt.figure(figsize=figsize)

    # srange=(0, 1.5),
    if len(obs_text) > 30:
        obs_text = obs_text[:25] + "..."

    td = TaylorDiagram(
        obs_std, fig=fig, rect=111, label=obs_text, normalize_std=normalize_std
    )
    contours = td.add_contours(levels=8, colors="0.5", linestyles="dotted")
    plt.clabel(contours, inline=1, fontsize=10, fmt="%.2f")

    if isinstance(points, TaylorPoint):
        points = [points]
    for p in points:
        assert isinstance(p, TaylorPoint)
        m = "o" if p.marker is None else p.marker
        ms = "6" if p.marker_size is None else p.marker_size
        std = p.std / p.obs_std if normalize_std else p.std
        td.add_sample(std, p.cc, marker=m, ms=ms, ls="", label=p.name)
        # marker=f"${1}$",
        # td.add_sample(0.2, 0.8, marker="+", ms=15, mew=1.2, ls="", label="m2")
    td.add_grid()
    fig.legend(
        td.samplePoints,
        [p.get_label() for p in td.samplePoints],
        numpoints=1,
        prop=dict(size="medium"),
        loc="upper right",
    )
    fig.suptitle(title, size="x-large")

    # prevent the plot from being displayed, since it is also displayed by the returned object
    plt.close()
    return fig
