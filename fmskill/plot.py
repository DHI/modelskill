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
    binsize: float = None,
    nbins: int = 20,
    show_points: Union[bool, int, str] = None,
    show_hist: bool = True,
    backend: str = "matplotlib",
    figsize: List[float] = (8, 8),
    xlim: List[float] = None,
    ylim: List[float] = None,
    reg_method: str = "ols",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
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
    binsize : float, optional
        the size of each bin in the 2d histogram, by default None
    nbins : int, optional
        number of bins (if binsize is not given), by default 20
    show_points : (bool, int, str), optional
        Should the scatter points be displayed?
        None means: only show points if fewer than threshold, by default None.
        float: fraction of points to show on plot from 0 to 1. eg 0.5 shows 50% of the points.
        int: if 'n' (int) given, then 'n' points will be displayed, randomly selected.
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

    if len(x) != len(y):
        raise ValueError("x & y are not of equal length")

    x_sample = x
    y_sample = y
    if show_points is None:
        # If nothing given, and more than 10k points, nothing will be shown
        if len(x) < 1e4:
            show_points = True
        else:
            show_points = False
    elif type(show_points) == float:
        if show_points < 0 or show_points > 1:
            raise TypeError(" `show_points` fraction must be in [0,1]")
        else:
            np.random.seed(20)
            ran_index = np.random.choice(range(len(x)), int(len(x) * show_points))
            x_sample = x[ran_index]
            y_sample = y[ran_index]
    # if show_points is an int
    elif type(show_points) == int:
        np.random.seed(20)
        ran_index = np.random.choice(range(len(x)), show_points)
        x_sample = x[ran_index]
        y_sample = y[ran_index]
    elif type(show_points) == bool:
        pass
    else:
        raise TypeError(" `show_points` must be either bool, int or float")

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xymin = min([xmin, ymin])
    xymax = max([xmax, ymax])

    if xlim is None:
        xlim = [xymin, xymax]

    if ylim is None:
        ylim = [xymin, xymax]

    if binsize is None:
        binsize = (xmax - xmin) / nbins
    else:
        nbins = int((xmax - xmin) / binsize)

    xq = np.quantile(x, q=np.linspace(0, 1, num=nbins))
    yq = np.quantile(y, q=np.linspace(0, 1, num=nbins))

    # linear fit
    slope, intercept = _linear_regression(obs=x, model=y, reg_method=reg_method)

    if intercept < 0:
        sign = ""
    else:
        sign = "+"
    reglabel = f"Fit: y={slope:.2f}x{sign}{intercept:.2f}"

    if backend == "matplotlib":

        plt.figure(figsize=figsize)
        plt.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], label="1:1", c="blue")
        plt.plot(xq, yq, label="Q-Q", c="gray")
        plt.plot(
            x,
            intercept + slope * x,
            "r",
            label=reglabel,
        )
        if show_hist:
            plt.hist2d(x, y, bins=nbins, cmin=0.01, **kwargs)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axis("square")
        plt.xlim(xlim)
        plt.ylim(ylim)
        if show_hist:
            cbar = plt.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label("# points")
        if show_points:
            plt.scatter(
                x_sample, y_sample, c="0.25", s=20, alpha=0.5, marker=".", label=None
            )
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
            go.Scatter(x=xq, y=yq, name="Q-Q", mode="lines", line=dict(color="gray")),
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
                    x=x_sample,
                    y=y_sample,
                    mode="markers",
                    name="Data",
                    marker=dict(color="black"),
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
