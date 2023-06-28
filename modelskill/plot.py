from typing import List, Tuple, Union, Optional, Sequence
import warnings
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from collections import namedtuple
from scipy import interpolate

import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator

from .model.point import PointModelResult
from .model.track import TrackModelResult
from .observation import Observation, PointObservation, TrackObservation
from .metrics import _linear_regression
from .plot_taylor import TaylorDiagram
import modelskill.settings as settings
from .settings import options, register_option
from .observation import unit_display_name
from .metrics import metric_has_units


register_option("plot.scatter.points.size", 20, validator=settings.is_positive)
register_option("plot.scatter.points.alpha", 0.5, validator=settings.is_between_0_and_1)
register_option("plot.scatter.points.label", "", validator=settings.is_str)
register_option("plot.scatter.quantiles.marker", "X", validator=settings.is_str)
register_option(
    "plot.scatter.quantiles.markersize", 3.5, validator=settings.is_positive
)
register_option(
    "plot.scatter.quantiles.color",
    "darkturquoise",
    validator=settings.is_tuple_list_or_str,
)
register_option("plot.scatter.quantiles.label", "Q-Q", validator=settings.is_str)
register_option(
    "plot.scatter.quantiles.markeredgecolor",
    (0, 0, 0, 0.4),
    validator=settings.is_tuple_list_or_str,
)
register_option(
    "plot.scatter.quantiles.markeredgewidth", 0.5, validator=settings.is_positive
)
register_option("plot.scatter.quantiles.kwargs", {}, validator=settings.is_dict)
register_option("plot.scatter.oneone_line.label", "1:1", validator=settings.is_str)
register_option(
    "plot.scatter.oneone_line.color", "blue", validator=settings.is_tuple_list_or_str
)
register_option("plot.scatter.legend.kwargs", {}, validator=settings.is_dict)
register_option(
    "plot.scatter.reg_line.kwargs", {"color": "r"}, validator=settings.is_dict
)
register_option(
    "plot.scatter.legend.bbox",
    {
        "facecolor": "blue",
        "edgecolor": "k",
        "boxstyle": "round",
        "alpha": 0.05,
    },
    validator=settings.is_dict,
)
# register_option("plot.scatter.table.show", False, validator=settings.is_bool)
register_option("plot.scatter.legend.fontsize", 12, validator=settings.is_positive)

# TODO: Auto-implement
# still requires plt.rcParams.update(modelskill.settings.get_option('plot.rcParams'))
register_option("plot.rcParams", {}, settings.is_dict)  # still have to


def _get_ax(ax=None, figsize=None):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _get_fig_ax(ax=None, figsize=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.gcf()
    return fig, ax


def sample_points(
    x: np.ndarray, y: np.ndarray, include: Optional[Union[bool, int, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points to be plotted

    Parameters
    ----------
    x: np.ndarray, 1d
    y: np.ndarray, 1d
    include: bool, int or float, optional
        default is subset the data to 50k points

    Returns
    -------
    np.ndarray, np.ndarray
        x and y arrays with sampled points
    """

    assert len(x) == len(y), "x and y must have same length"

    if include is True:
        return x, y

    if include is None:
        if len(x) < 5e4:
            return x, y
        else:
            include = 50000
            warnings.warn(
                message=f"Showing only {include} points in plot. Set `include` to True to show all points."
            )
    else:
        if not isinstance(include, (bool, int, float)):
            raise TypeError(f"'subset' must be bool, int or float, not {type(include)}")

    if include is False:
        return np.array([]), np.array([])

    if isinstance(include, float):
        if not 0 <= include <= 1:
            raise ValueError("`include` fraction must be in [0,1]")

        n_samples = int(len(x) * include)
    elif isinstance(include, int):
        if include < 0:
            raise ValueError("`include` must be positive integer")
        if include > len(x):
            include = len(x)
        n_samples = include

    np.random.seed(20)  # TODO should this be a parameter?
    ran_index = np.random.choice(range(len(x)), n_samples, replace=False)
    x_sample = x[ran_index]
    y_sample = y[ran_index]

    return x_sample, y_sample


def quantiles_xy(
    x: np.ndarray, y: np.ndarray, quantiles: Union[int, Sequence[float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate quantiles of x and y

    Parameters
    ----------
    x: np.ndarray, 1d
    y: np.ndarray, 1d
    q: int, Sequence[float]
        quantiles to calculate

    Returns
    -------
    np.ndarray, np.ndarray
        x and y arrays with quantiles
    """

    if quantiles is None:
        if len(x) >= 3000:
            quantiles = 1000
        elif len(x) >= 300:
            quantiles = 100
        else:
            quantiles = 10

    if not isinstance(quantiles, (int, Sequence)):
        raise TypeError("quantiles must be an int or sequence of floats")

    q = np.linspace(0, 1, num=quantiles) if isinstance(quantiles, int) else quantiles
    return np.quantile(x, q=q), np.quantile(y, q=q)


def _scatter_matplotlib(
    *,
    x,
    y,
    x_sample,
    y_sample,
    z,
    xq,
    yq,
    x_trend,
    show_density,
    show_points,
    show_hist,
    norm,
    nbins_hist,
    intercept,
    slope,
    xlabel,
    ylabel,
    figsize,
    xlim,
    ylim,
    title,
    skill_df,
    units,
    **kwargs,
) -> Axes:
    _, ax = plt.subplots(figsize=figsize)

    plt.plot(
        [xlim[0], xlim[1]],
        [xlim[0], xlim[1]],
        label=options.plot.scatter.oneone_line.label,
        c=options.plot.scatter.oneone_line.color,
        zorder=3,
    )

    if show_points is None or show_points:
        if show_density:
            c = z
        else:
            c = "0.25"
        plt.scatter(
            x_sample,
            y_sample,
            c=c,
            s=options.plot.scatter.points.size,
            alpha=options.plot.scatter.points.alpha,
            marker=".",
            label=options.plot.scatter.points.label,
            zorder=1,
            norm=norm,
            **kwargs,
        )
    plt.plot(
        xq,
        yq,
        options.plot.scatter.quantiles.marker,
        label=options.plot.scatter.quantiles.label,
        c=options.plot.scatter.quantiles.color,
        zorder=4,
        markeredgecolor=options.plot.scatter.quantiles.markeredgecolor,
        markeredgewidth=options.plot.scatter.quantiles.markeredgewidth,
        markersize=options.plot.scatter.quantiles.markersize,
        **settings.get_option("plot.scatter.quantiles.kwargs"),
    )

    plt.plot(
        x_trend,
        intercept + slope * x_trend,
        **settings.get_option("plot.scatter.reg_line.kwargs"),
        label=_reglabel(slope=slope, intercept=intercept),
        zorder=2,
    )

    if show_hist:
        plt.hist2d(x, y, bins=nbins_hist, cmin=0.01, zorder=0.5, **kwargs)

    plt.legend(**settings.get_option("plot.scatter.legend.kwargs"))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis("square")
    plt.xlim([xlim[0], xlim[1]])
    plt.ylim([ylim[0], ylim[1]])
    plt.minorticks_on()
    plt.grid(which="both", axis="both", linewidth="0.2", color="k", alpha=0.6)
    max_cbar = None
    if show_hist or (show_density and show_points):
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        ticks = cbar.ax.get_yticks()
        max_cbar = ticks[-1]
        cbar.set_label("# points")
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title(title)
    # Add skill table
    if skill_df is not None:
        df = skill_df.df
        assert isinstance(df, pd.DataFrame)
        _plot_summary_table(df, units, max_cbar=max_cbar)
    return ax


def _scatter_plotly(
    *,
    x,
    y,
    x_sample,
    y_sample,
    z,
    xq,
    yq,
    x_trend,
    show_density,
    show_points,
    norm,  # TODO not used by plotly, remove or keep for consistency?
    show_hist,
    nbins_hist,
    intercept,
    slope,
    xlabel,
    ylabel,
    figsize,  # TODO not used by plotly, remove or keep for consistency?
    xlim,
    ylim,
    title,
    skill_df,  # TODO implement
    units,  # TODO implement
    **kwargs,
):

    import plotly.graph_objects as go

    data = [
        go.Scatter(x=xlim, y=xlim, name="1:1", mode="lines", line=dict(color="blue")),
    ]

    regression_line = go.Scatter(
        x=x_trend,
        y=intercept + slope * x_trend,
        name=_reglabel(slope=slope, intercept=intercept),
        mode="lines",
        line=dict(color="red"),
    )
    data.append(regression_line)

    if show_hist:
        data.append(
            go.Histogram2d(
                x=x,
                y=y,
                nbinsx=nbins_hist,
                nbinsy=nbins_hist,
                colorscale=[
                    [0.0, "rgba(0,0,0,0)"],
                    [0.1, "purple"],
                    [0.5, "green"],
                    [1.0, "yellow"],
                ],
                colorbar=dict(title="# of points"),
            )
        )

    if show_points is None or show_points:
        if show_density:
            c = z
            cbar = dict(thickness=20, title="# of points")
        else:
            c = "black"
            cbar = None
        data.append(
            go.Scatter(
                x=x_sample,
                y=y_sample,
                mode="markers",
                name="Data",
                marker=dict(color=c, opacity=0.5, size=3.0, colorbar=cbar),
            )
        )
    data.append(
        go.Scatter(
            x=xq,
            y=yq,
            name=options.plot.scatter.quantiles.label,
            mode="markers",
            marker_symbol="x",
            marker_color=options.plot.scatter.quantiles.color,
            marker_line_color="midnightblue",
            marker_line_width=0.6,
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


def _reglabel(slope: float, intercept: float) -> str:
    sign = "" if intercept < 0 else "+"
    return f"Fit: y={slope:.2f}x{sign}{intercept:.2f}"


def _get_bins(
    bins: Union[int, float, Sequence[float]], xymin, xymax
):  # TODO return type

    assert xymax >= xymin
    xyspan = xymax - xymin

    if isinstance(bins, int):
        nbins_hist = bins
        binsize = xyspan / nbins_hist
    elif isinstance(bins, float):
        binsize = bins
        nbins_hist = int(xyspan / binsize)
    elif isinstance(bins, Sequence):
        binsize = bins
        nbins_hist = bins
    else:
        raise TypeError("bins must be an int, float or sequence")

    return nbins_hist, binsize


def scatter(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: Union[int, float, List[int], List[float]] = 20,
    quantiles: Optional[Union[int, List[float]]] = None,
    fit_to_quantiles: bool = False,
    show_points: Optional[Union[bool, int, float]] = None,
    show_hist: Optional[bool] = None,
    show_density: Optional[bool] = None,
    norm: Optional[colors.Normalize] = None,
    backend: str = "matplotlib",
    figsize: Tuple[float, float] = (8, 8),
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    reg_method: str = "ols",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    skill_df: Optional[pd.DataFrame] = None,
    units: Optional[str] = "",
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
    fit_to_quantiles: bool, optional, by default False
        by default the regression line is fitted to all data, if True, it is fitted to the quantiles
        which can be useful to represent the extremes of the distribution
    show_points : (bool, int, float), optional
        Should the scatter points be displayed?
        None means: show all points if fewer than 1e4, otherwise show 1e4 sample points, by default None.
        float: fraction of points to show on plot from 0 to 1. eg 0.5 shows 50% of the points.
        int: if 'n' (int) given, then 'n' points will be displayed, randomly selected.
    show_hist : bool, optional
        show the data density as a 2d histogram, by default None
    show_density: bool, optional
        show the data density as a colormap of the scatter, by default None. If both `show_density` and `show_hist`
        are None, then `show_density` is used by default.
        for binning the data, the previous kword `bins=Float` is used
    norm : matplotlib.colors.Normalize
        colormap normalization
        If None, defaults to matplotlib.colors.PowerNorm(vmin=1,gamma=0.5)
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
    skill_df : dataframe, optional
        dataframe with skill (stats) results to be added to plot, by default None
    units : str, optional
        user default units to override default units, eg 'metre', by default None
    kwargs
    """
    if show_hist is None and show_density is None:
        # Default: points density
        show_density = True

    if len(x) != len(y):
        raise ValueError("x & y are not of equal length")

    if norm is None:
        # Default: PowerNorm with gamma of 0.5
        norm = colors.PowerNorm(vmin=1, gamma=0.5)

    x_sample, y_sample = sample_points(x, y, show_points)
    xq, yq = quantiles_xy(x, y, quantiles)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xymin = min([xmin, ymin])
    xymax = max([xmax, ymax])

    nbins_hist, binsize = _get_bins(bins, xymin=xymin, xymax=xymax)

    if xlim is None:
        xlim = (xymin - binsize, xymax + binsize)

    if ylim is None:
        ylim = (xymin - binsize, xymax + binsize)

    x_trend = np.array([xlim[0], xlim[1]])

    if show_hist and show_density:
        raise TypeError(
            "if `show_hist=True` then `show_density` must be either `False` or `None`"
        )

    z = None
    if show_density and len(x_sample) > 0:
        if not isinstance(bins, (float, int)):
            raise TypeError(
                "if `show_density=True` then bins must be either float or int"
            )

        # calculate density data
        z = __scatter_density(x_sample, y_sample, binsize=binsize)
        idx = z.argsort()
        # Sort data by colormaps
        x_sample, y_sample, z = x_sample[idx], y_sample[idx], z[idx]
        # scale Z by sample size
        z = z * len(x) / len(x_sample)

    # linear fit
    if fit_to_quantiles:
        slope, intercept = _linear_regression(obs=xq, model=yq, reg_method=reg_method)
    else:
        slope, intercept = _linear_regression(obs=x, model=y, reg_method=reg_method)

    PLOTTING_BACKENDS = {
        "matplotlib": _scatter_matplotlib,
        "plotly": _scatter_plotly,
    }

    if backend not in PLOTTING_BACKENDS:
        raise ValueError(f"backend must be one of {list(PLOTTING_BACKENDS.keys())}")

    return PLOTTING_BACKENDS[backend](
        x=x,
        y=y,
        x_sample=x_sample,
        y_sample=y_sample,
        z=z,
        xq=xq,
        yq=yq,
        x_trend=x_trend,
        show_density=show_density,
        norm=norm,
        show_points=show_points,
        show_hist=show_hist,
        nbins_hist=nbins_hist,
        intercept=intercept,
        slope=slope,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        xlim=xlim,
        ylim=ylim,
        title=title,
        skill_df=skill_df,
        units=units,
        **kwargs,
    )


def plot_temporal_coverage(
    obs=None,
    mod=None,
    *,
    limit_to_model_period=True,
    marker="_",
    ax=None,
    figsize=None,
    title=None,
):
    """Plot graph showing temporal coverage for all observations and models

    Parameters
    ----------
    obs : List[Observation], optional
        Show observation(s) as separate lines on plot
    mod : List[ModelResult], optional
        Show model(s) as separate lines on plot, by default None
    limit_to_model_period : bool, optional
        Show temporal coverage only for period covered
        by the model, by default True
    marker : str, optional
        plot marker for observations, by default "_"
    ax: matplotlib.axes, optional
        Adding to existing axis, instead of creating new fig
    figsize : Tuple(float, float), optional
        size of figure, by default (7, 0.45*n_lines)
    title: str, optional
        plot title, default empty

    See Also
    --------
    plot_spatial_overview

    Returns
    -------
    <matplotlib.axes>

    Examples
    --------
    >>> import modelskill as ms
    >>> o1 = ms.PointObservation('HKNA_Hm0.dfs0', item=0, x=4.2420, y=52.6887, name="HKNA")
    >>> o2 = ms.TrackObservation("Alti_c2_Dutch.dfs0", item=3, name="c2")
    >>> mr1 = ModelResult('HKZN_local_2017_DutchCoast.dfsu', name='SW_1', item=0)
    >>> mr2 = ModelResult('HKZN_local_2017_DutchCoast_v2.dfsu', name='SW_2', item=0)
    >>> ms.plot_temporal_coverage([o1, o2], [mr1, mr2])
    >>> ms.plot_temporal_coverage([o1, o2], mr2, limit_to_model_period=False)
    >>> ms.plot_temporal_coverage(o2, [mr1, mr2], marker=".")
    >>> ms.plot_temporal_coverage(mod=[mr1, mr2], figsize=(5,3))
    """
    obs = [] if obs is None else list(obs) if isinstance(obs, Sequence) else [obs]
    mod = [] if mod is None else list(mod) if isinstance(mod, Sequence) else [mod]

    n_lines = len(obs) + len(mod)
    if figsize is None:
        ysize = max(2.0, 0.45 * n_lines)
        figsize = (7, ysize)

    fig, ax = _get_fig_ax(ax=ax, figsize=figsize)
    y = np.repeat(0.0, 2)
    labels = []

    if len(mod) > 0:
        for mr in mod:
            y += 1.0
            plt.plot([mr.start_time, mr.end_time], y)
            labels.append(mr.name)

    for o in obs:
        y += 1.0
        plt.plot(o.time, y[0] * np.ones(len(o.time)), marker, markersize=5)
        labels.append(o.name)

    if len(mod) > 0 and limit_to_model_period:
        mr = mod[0]  # take first model
        plt.xlim([mr.start_time, mr.end_time])

    plt.yticks(np.arange(n_lines) + 1, labels)
    if len(mod) > 0:
        for j in range(len(mod)):
            ax.get_yticklabels()[j].set_fontstyle("italic")
            ax.get_yticklabels()[j].set_weight("bold")
            # set_color("#004165")
    fig.autofmt_xdate()

    if title:
        ax.set_title(title)
    return ax


def plot_spatial_overview(
    obs: List[Observation],
    mod=None,
    ax=None,
    figsize: Tuple = None,
    title: str = None,
):
    """Plot observation points on a map showing the model domain

    Parameters
    ----------
    obs: list[Observation]
        List of observations to be shown on map
    mod : Union[ModelResult, mikeio.GeometryFM], optional
        Model domain to be shown as outline
    ax: matplotlib.axes, optional
        Adding to existing axis, instead of creating new fig
    figsize : (float, float), optional
        figure size, by default None
    title: str, optional
        plot title, default empty

    See Also
    --------
    plot_temporal_coverage

    Returns
    -------
    <matplotlib.axes>

    Examples
    --------
    >>> import modelskill as ms
    >>> o1 = ms.PointObservation('HKNA_Hm0.dfs0', item=0, x=4.2420, y=52.6887, name="HKNA")
    >>> o2 = ms.TrackObservation("Alti_c2_Dutch.dfs0", item=3, name="c2")
    >>> mr1 = ModelResult('HKZN_local_2017_DutchCoast.dfsu', name='SW_1', item=0)
    >>> mr2 = ModelResult('HKZN_local_2017_DutchCoast_v2.dfsu', name='SW_2', item=0)
    >>> ms.plot_spatial_overview([o1, o2], [mr1, mr2])
    """
    obs = [] if obs is None else list(obs) if isinstance(obs, Sequence) else [obs]
    mod = [] if mod is None else list(mod) if isinstance(mod, Sequence) else [mod]

    ax = _get_ax(ax=ax, figsize=figsize)
    offset_x = 1  # TODO: better default

    for m in mod:
        # TODO: support Gridded ModelResults
        if isinstance(m, (PointModelResult, TrackModelResult)):
            raise ValueError(
                f"Model type {type(m)} not supported. Only DfsuModelResult and mikeio.GeometryFM supported!"
            )
        if hasattr(m, "data") and hasattr(m.data, "geometry"):
            # mod_name = m.name  # TODO: better support for multiple models
            m = m.data.geometry
        if hasattr(m, "node_coordinates"):
            xn = m.node_coordinates[:, 0]
            offset_x = 0.02 * (max(xn) - min(xn))
        m.plot.outline(ax=ax)

    for o in obs:
        if isinstance(o, PointObservation):
            ax.scatter(x=o.x, y=o.y, marker="x")
            ax.annotate(o.name, (o.x + offset_x, o.y))
        elif isinstance(o, TrackObservation):
            if o.n_points < 10000:
                ax.scatter(x=o.x, y=o.y, c=o.values, marker=".", cmap="Reds")
            else:
                print(f"{o.name}: Too many points to plot")
                # TODO: group by lonlat bin or sample randomly
        else:
            raise ValueError(
                f"Could not show observation {o}. Only PointObservation and TrackObservation supported."
            )

    if not title:
        title = "Spatial coverage"
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


def __hist2d(x, y, binsize):
    """Calculates 2D histogram (gridded) of data.

    Parameters
    ----------
    x: np.array
        X values e.g model values, must be same length as y
    y: np.array
        Y values e.g observation values, must be same length as x
    binsize: float, optional
        2D histogram (bin) resolution, by default = 0.1

    Returns
    ----------
    histodata: np.array
        2D-histogram data
    cxy: np.array
        Center points of the histogram bins
    exy: np.array
        Edges of the histogram bins
    """
    # Make linear-grid for interpolation
    minxy = min(min(x), min(y)) - binsize
    maxxy = max(max(x), max(y)) + binsize
    # Center points of the bins
    cxy = np.arange(minxy, maxxy, binsize)
    # Edges of the bins
    exy = np.arange(minxy - binsize * 0.5, maxxy + binsize * 0.5, binsize)
    if exy[-1] <= cxy[-1]:
        # sometimes, given the bin size, the edges array ended before (left side) of the bins-center array
        # in such case, and extra half-bin is added at the end
        exy = np.arange(minxy - binsize * 0.5, maxxy + binsize, binsize)

    # Calculate 2D histogram
    histodata, _, _ = np.histogram2d(x, y, [exy, exy])

    # Histogram values
    hist = []
    for j in range(len(cxy)):
        for i in range(len(cxy)):
            hist.append(histodata[i, j])

    return hist, cxy


def __scatter_density(x, y, binsize: float = 0.1, method: str = "linear"):
    """Interpolates scatter data on a 2D histogram (gridded) based on data density.

    Parameters
    ----------
    x: np.array
        X values e.g model values, must be same length as y
    y: np.array
        Y values e.g observation values, must be same length as x
    binsize: float, optional
        2D histogram (bin) resolution, by default = 0.1
    method: str, optional
        Scipy griddata interpolation method, by default 'linear'

    Returns
    ----------
    Z_grid: np.array
        Array with the colors based on histogram density
    """

    hist, cxy = __hist2d(x, y, binsize)

    # Grid-data
    xg, yg = np.meshgrid(cxy, cxy)
    xg = xg.ravel()
    yg = yg.ravel()

    ## Interpolate histogram density data to scatter data
    Z_grid = interpolate.griddata((xg, yg), hist, (x, y), method=method)

    # Replace negative values (should there be some) in case of 'cubic' interpolation
    Z_grid[(Z_grid < 0)] = 0

    return Z_grid


def _format_skill_line(
    series: pd.Series,
    units: str,
    precision: int,
) -> str:

    name = series.name

    item_unit = " "

    if name == "n":
        fvalue = series.values[0]
    else:
        if metric_has_units(metric=name):
            # if statistic has dimensions, then add units
            item_unit = unit_display_name(units)

        rounded_value = np.round(series.values[0], precision)
        fmt = f".{precision}f"
        fvalue = f"{rounded_value:{fmt}}"

    name = series.name.upper()

    return f"{name}", " =  ", f"{fvalue} {item_unit}"


def format_skill_df(df: pd.DataFrame, units: str, precision: int = 2) -> List[str]:

    # remove model and variable columns if present, i.e. keep all other columns
    df.drop(["model", "variable"], axis=1, errors="ignore", inplace=True)

    # loop over series in dataframe, (columns)
    lines = [_format_skill_line(df[col], units, precision) for col in list(df.columns)]

    return np.array(lines)


def _plot_summary_border(
    figure_transform,
    x0,
    y0,
    dx,
    dy,
    borderpad=0.01,
) -> None:

    ## Load settings
    bbox_kwargs = {}
    bbox_kwargs.update(settings.get_option("plot.scatter.legend.bbox"))
    if (
        "boxstyle" in bbox_kwargs and "pad" not in bbox_kwargs["boxstyle"]
    ):  # default padding results in massive bbox
        bbox_kwargs["boxstyle"] = bbox_kwargs["boxstyle"] + f",pad={borderpad}"
    else:
        bbox_kwargs["boxstyle"] = f"square,pad={borderpad}"
    lgkw = settings.get_option("plot.scatter.legend.kwargs")
    if "edgecolor" in lgkw:
        bbox_kwargs["edgecolor"] = lgkw["edgecolor"]

    ## Define rectangle
    bbox = patches.FancyBboxPatch(
        (x0 - borderpad, y0 - borderpad),
        dx + borderpad * 2,
        dy + borderpad * 2,
        transform=figure_transform,
        clip_on=False,
        **bbox_kwargs,
    )

    plt.gca().add_patch(bbox)


def _plot_summary_table(
    df: pd.DataFrame, units: str, max_cbar: Optional[float] = None
) -> None:

    lines = format_skill_df(df, units)
    text_ = ["\n".join(lines[:, i]) for i in range(lines.shape[1])]

    if max_cbar is None:
        x = 0.93
    elif max_cbar < 1e3:
        x = 0.99
    elif max_cbar < 1e4:
        x = 1.01
    elif max_cbar < 1e5:
        x = 1.03
    elif max_cbar < 1e6:
        x = 1.05
    else:
        # When more than 1e6 samples, matplotlib changes to scientific notation
        x = 0.97

    fig = plt.gcf()
    figure_transform = fig.transFigure.get_affine()

    # General text settings
    txt_settings = dict(
        fontsize=options.plot.scatter.legend.fontsize,
    )

    # Column 1
    text_columns = []
    dx = 0
    for ti in text_:
        text_col_i = fig.text(x + dx, 0.6, ti, **txt_settings)
        ## Render, and get width
        plt.draw()
        dx = (
            dx
            + figure_transform.inverted().transform(
                [text_col_i.get_window_extent().bounds[2], 0]
            )[0]
        )
        text_columns.append(text_col_i)

    # Plot border
    ## Define coordintes
    x0, y0 = figure_transform.inverted().transform(
        text_columns[0].get_window_extent().bounds[0:2]
    )
    _, dy = figure_transform.inverted().transform(
        (0, text_columns[0].get_window_extent().bounds[3])
    )

    _plot_summary_border(figure_transform, x0, y0, dx, dy)
