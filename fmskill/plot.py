from typing import List, Tuple, Union
import warnings
import numpy as np
from collections import namedtuple

import matplotlib.pyplot as plt

from mikeio import Dfsu

from .observation import Observation, PointObservation, TrackObservation
from .metrics import _linear_regression
from .plot_taylor import TaylorDiagram


def scatter(
    x,
    y,
    *,
    bins: Union[int, float, List[int], List[float]] = 20,
    quantiles: Union[int, List[float]] = None,
    show_points: bool = None,
    show_hist: bool = True,
    backend: str = "matplotlib",
    figsize: List[float] = (8, 8),
    xlim: List[float] = None,
    ylim: List[float] = None,
    reg_method: str = "ols",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    binsize: float = None,
    nbins: int = None,
    **kwargs,
):
    """Scatter plot showing compared data: observation vs modelled
    Optionally, with density histogram.

    Parameters
    ----------
    x: np.array
        X values e.g model values, must be same length as y
    y: np.array
        Y values e.g observation values, must be same length as x
    bins: (int, float, sequence), optional
        bins for the 2D histogram on the background. By default 20 bins.
        if int, represents the number of bins of 2D
        if float, represents the bin size
        if sequence (list of int or float), represents the bin edges
    quantiles: (int, sequence), optional
        number of quantiles for QQ-plot, by default None and will depend on the scatter data length (10, 100 or 1000)
        if int, this is the number of points
        if sequence (list of floats), represents the desired quantiles (from 0 to 1)
    show_points : bool, optional
        Should the scatter points be displayed?
        None means: only show points if fewer than threshold, by default None
    show_hist : bool, optional
        show the data density as a a 2d histogram, by default True
    backend : str, optional
        use "plotly" (interactive) or "matplotlib" backend, by default "matplotlib"
    figsize : tuple, optional
        width and height of the figure, by default (8, 8)
    xlim : tuple, optional
        plot range for the observation (xmin, xmax), by default None
    ylim : tuple, optional
        plot range for the model (ymin, ymax), by default None
    reg_method : str, optional
        method for determining the regression line
        "ols" : ordinary least squares regression
        "odr" : orthogonal distance regression,
        by default "ols"
    title : str, optional
        plot title, by default None
    xlabel : str, optional
        x-label text on plot, by default None
    ylabel : str, optional
        y-label text on plot, by default None
    kwargs
    """

    if (binsize is not None) or (nbins is not None):
        warnings.warn(
            "`binsize` and `nbins` are deprecated and will be removed soon, use `bins` instead",
        )
        binsize_aux = binsize
        nbins_aux = nbins
    else:
        binsize_aux = None
        nbins_aux = None

    if len(x) != len(y):
        raise ValueError("x & y are not of equal length")

    if show_points is None:
        show_points = len(x) < 1e4

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xymin = min([xmin, ymin])
    xymax = max([xmax, ymax])

    if xlim is None:
        xlim = [xymin, xymax]

    if ylim is None:
        ylim = [xymin, xymax]

    if quantiles is None:
        if len(x) >= 3000:
            quantiles = 1000
        elif len(x) >= 300:
            quantiles = 100
        else:
            quantiles = 10

    if type(bins) == int:
        nbins_hist = bins
        binsize = int((xymax - xymin) / nbins_hist)
    elif type(bins) == float:
        binsize = bins
        nbins_hist = int((xymax - xymin) / binsize)
    else:
        # Then bins = Sequence
        binsize = bins
        nbins_hist = bins

    # Check deprecated kwords; Remove this verification in future release
    if (binsize_aux is not None) or (nbins_aux is not None):
        if binsize_aux is None:
            binsize = (xmax - xmin) / nbins_aux
            nbins_hist = nbins_aux
        else:
            nbins_hist = int((xmax - xmin) / binsize_aux)
    # Remove previous piece of code when nbins and bin_size are deprecated.

    if type(quantiles) == int:
        xq = np.quantile(x, q=np.linspace(0, 1, num=quantiles))
        yq = np.quantile(y, q=np.linspace(0, 1, num=quantiles))
    else:
        # if not an int nor None, it must be a squence of floats
        xq = np.quantile(x, q=quantiles)
        yq = np.quantile(y, q=quantiles)

    # linear fit
    slope, intercept = _linear_regression(obs=x, model=y, reg_method=reg_method)

    if intercept < 0:
        sign = ""
    else:
        sign = "+"
    reglabel = f"Fit: y={slope:.2f}x{sign}{intercept:.2f}"

    if backend == "matplotlib":

        plt.figure(figsize=figsize)
        plt.plot(
            [xlim[0], xlim[1]],
            [xlim[0], xlim[1]],
            label="1:1",
            c="blue",
            zorder=3,
        )
        if show_points:
            plt.scatter(
                x,
                y,
                c="0.25",
                s=20,
                alpha=0.5,
                marker=".",
                label=None,
                zorder=1,
            )
        plt.plot(
            xq,
            yq,
            "o",
            label="Q-Q",
            c="darkturquoise",
            markeredgecolor=(0, 0, 0, 0.4),
            zorder=2,
        )
        plt.plot(
            x,
            intercept + slope * x,
            "r",
            label=reglabel,
        )
        if show_hist:
            plt.hist2d(x, y, bins=nbins_hist, cmin=0.01, zorder=0.5, **kwargs)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axis("square")
        plt.xlim(xlim)
        plt.ylim(ylim)
        if show_hist:
            cbar = plt.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label("# points")

        plt.title(title)

    elif backend == "plotly":  # pragma: no cover
        import plotly.graph_objects as go

        data = [
            go.Scatter(
                x=x,
                y=intercept + slope * x,
                name=reglabel,
                mode="lines",
                line=dict(color="red"),
            ),
            go.Scatter(
                x=xlim, y=xlim, name="1:1", mode="lines", line=dict(color="blue")
            ),
        ]

        if show_hist:
            data.append(
                go.Histogram2d(
                    x=x,
                    y=y,
                    xbins=dict(size=binsize),
                    ybins=dict(size=binsize),
                    colorscale=[
                        [0.0, "rgba(0,0,0,0)"],
                        [0.1, "purple"],
                        [0.5, "green"],
                        [1.0, "yellow"],
                    ],
                )
            )

        if show_points:
            data.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    name="Data",
                    marker=dict(color="black", opacity=0.5, size=3.0),
                )
            )
        data.append(
            go.Scatter(
                x=xq, y=yq, name="Q-Q", mode="markers", line=dict(color="darkturquoise")
            )
        )

        defaults = {"width": 600, "height": 600}
        defaults = {**defaults, **kwargs}

        layout = layout = go.Layout(
            legend=dict(x=0.01, y=0.99),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            title=dict(text=title, xanchor="center", yanchor="top", x=0.5, y=0.9),
            yaxis_title=ylabel,
            xaxis_title=xlabel,
            **defaults,
        )

        fig = go.Figure(data=data, layout=layout)
        fig.update_xaxes(range=xlim)
        fig.update_yaxes(range=ylim)
        fig.show()  # Should this be here

    else:

        raise ValueError(f"Plotting backend: {backend} not supported")


def plot_observation_positions(
    dfs: Dfsu, observations: List[Observation], figsize: Tuple = None, title=None
):
    """Plot observation points on a map showing the model domain

    Parameters
    ----------
    dfs: Dfsu
        model object
    observations: list
        Observation collection
    figsize : (float, float), optional
        figure size, by default None
    title: str, optional
        plot title, default empty
    """

    xn = dfs.node_coordinates[:, 0]
    offset_x = 0.02 * (max(xn) - min(xn))
    ax = dfs.plot(plot_type="outline_only", figsize=figsize)
    for obs in observations:
        if isinstance(obs, PointObservation):
            ax.scatter(x=obs.x, y=obs.y, marker="x")
            ax.annotate(obs.name, (obs.x + offset_x, obs.y))
        elif isinstance(obs, TrackObservation):
            if obs.n_points < 10000:
                ax.scatter(x=obs.x, y=obs.y, c=obs.values, marker=".", cmap="Reds")
            else:
                print("Too many points to plot")
                # TODO: group by lonlat bin
    if title:
        ax.set_title(title)
    return ax


TaylorPoint = namedtuple("TaylorPoint", "name obs_std std cc marker marker_size")


def taylor_diagram(
    obs_std,
    points,
    figsize=(7, 7),
    obs_text="Observations",
    normalize_std=False,
    title="Taylor diagram",
):
    if np.isscalar(figsize):
        figsize = (figsize, figsize)
    elif figsize[0] != figsize[1]:
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
