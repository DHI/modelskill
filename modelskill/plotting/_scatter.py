from __future__ import annotations
from typing import Literal, Optional, Sequence, Tuple, Callable, TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    import matplotlib.axes

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from scipy import interpolate
import pandas as pd

import modelskill.settings as settings
from modelskill.settings import options

from ..metrics import _linear_regression
from ._misc import quantiles_xy, sample_points, format_skill_table, _get_fig_ax


def scatter(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int | float = 120,
    quantiles: int | Sequence[float] | None = None,
    fit_to_quantiles: bool = False,
    show_points: bool | int | float | None = None,
    show_hist: Optional[bool] = None,
    show_density: Optional[bool] = None,
    norm: Optional[colors.Normalize] = None,
    backend: Literal["matplotlib", "plotly"] = "matplotlib",
    figsize: Tuple[float, float] = (8, 8),
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    reg_method: str | bool = "ols",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    skill_table: Optional[str | Sequence[str] | bool] = False,
    skill_scores: Mapping[str, float] | None = None,
    skill_score_unit: Optional[str] = "",
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:
    """Scatter plot tailored for model skill comparison.

    Scatter plot showing compared data: observation vs modelled
    Optionally, with density histogram or denisty color coding of points.

    Note: can be called directly but is often called through the plot
    accessor on a Comparer/CompararCollection cmp.plot.scatter()

    Parameters
    ----------
    x: np.array
        X values e.g model values, must be same length as y
    y: np.array
        Y values e.g observation values, must be same length as x
    bins: (int, float, sequence), optional
        bins for the 2D histogram on the background. By default 120 bins.
        if int, represents the number of bins of 2D
        if float, represents the bin size
        if sequence (list of int or float), represents the bin edges
    quantiles: (int, sequence), optional
        number of quantiles for QQ-plot, by default None and will depend on the scatter data length (10, 100 or 1000)
        if int, this is the number of points
        if sequence (list of floats), represents the desired quantiles (from 0 to 1)
    fit_to_quantiles: bool, optional
        by default the regression line is fitted to all data, if True, it is fitted to the quantiles
        which can be useful to represent the extremes of the distribution, by default False
    show_points : (bool, int, float), optional
        Should the scatter points be displayed?
        None means: show all points if fewer than 1e4, otherwise show 1e4 sample points, by default None.
        float: fraction of points to show on plot from 0 to 1. eg 0.5 shows 50% of the points.
        int: if 'n' (int) given, then 'n' points will be displayed, randomly selected.
    show_hist : bool, optional
        show the data density as a 2d histogram, by default None
    show_density: bool, optional
        show the data density as a colormap of the scatter, by default None. If both `show_density` and `show_hist`
        are None, then `show_density` is used by default. If number of points is less than 200, then `show_density`
        is False by default. For binning the data, the previous kword `bins=Float` is used
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
    reg_method : str or bool, optional
        method for determining the regression line
        "ols" : ordinary least squares regression
        "odr" : orthogonal distance regression,
        False : no regression line
        by default "ols"
    title : str, optional
        plot title, by default None
    xlabel : str, optional
        x-label text on plot, by default None
    ylabel : str, optional
        y-label text on plot, by default None
    skill_table: str, List[str], bool, optional
        calculate skill scores and show in box next to the plot,
        True will show default metrics, list of metrics will show
        these skill scores, by default False,
        Note: cannot be used together with skill_scores argument
    skill_scores : dict[str, float], optional
        dictionary with skill scores to be shown in box next to
        the plot, by default None
        Note: cannot be used together with skill_table argument
    skill_score_unit : str, optional
        unit for skill_scores, by default None
    ax : matplotlib.axes.Axes, optional
        axes to plot on, by default None
    **kwargs

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the scatter plot was drawn.

    Examples
    --------
    ```{python}
    import numpy as np
    import modelskill as ms

    x = np.linspace(0, 10, 1000)
    y = x + np.random.normal(size=1000)

    ms.plotting.scatter(x, y, skill_table=True)
    ```
    ```{python}
    ms.plotting.scatter(x, y, show_hist=True, bins=20, cmap="OrRd")
    ```
    ```{python}
    ms.plotting.scatter(x, y, quantiles=0, title="Hide quantiles")
    ```
    ```{python}
    ms.plotting.scatter(x, y, xlim=(0,4), ylim=(0,4), show_density=False)
    ```
    """

    if len(x) != len(y):
        raise ValueError("x & y are not of equal length")

    if norm is None:
        norm = colors.PowerNorm(vmin=1, gamma=0.5)

    x_sample, y_sample = sample_points(x, y, show_points)
    show_points = len(x_sample) > 0
    xq, yq = quantiles_xy(x, y, quantiles)

    if show_hist is None and show_density is None:
        # Default: points density
        show_density = len(x_sample) >= 200

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
        raise ValueError(
            "if `show_hist=True` then `show_density` must be either `False` or `None`"
        )

    if (show_points is False) and show_density:
        raise ValueError(
            "if `show_points=False` then `show_density` must be either "
            "`False` or `None`; density is a property (the color) of the points!"
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

    PLOTTING_BACKENDS: dict[str, Callable] = {
        "matplotlib": _scatter_matplotlib,
        "plotly": _scatter_plotly,
    }

    if backend not in PLOTTING_BACKENDS:
        raise ValueError(f"backend must be one of {list(PLOTTING_BACKENDS.keys())}")

    if skill_table:
        from modelskill import from_matched

        if skill_scores is not None:
            raise ValueError(
                "Cannot pass skill_scores and skill_table at the same time"
            )
        df = pd.DataFrame({"obs": x, "model": y})
        cmp = from_matched(df)
        metrics = None if skill_table is True else skill_table
        skill = cmp.skill(metrics=metrics)
        skill_scores = skill.to_dict("records")[0]

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
        reg_method=reg_method,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        xlim=xlim,
        ylim=ylim,
        title=title,
        skill_scores=skill_scores,
        skill_score_unit=skill_score_unit,
        fit_to_quantiles=fit_to_quantiles,
        ax=ax,
        **kwargs,
    )


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
    reg_method,
    xlabel,
    ylabel,
    figsize,
    xlim,
    ylim,
    title,
    skill_scores,
    skill_score_unit,
    fit_to_quantiles,
    ax,
    cmap=None,
    **kwargs,
) -> matplotlib.axes.Axes:
    fig, ax = _get_fig_ax(ax, figsize)

    if len(x) < 2:
        raise ValueError("Not enough data to plot. At least 2 points are required.")

    ax.plot(
        [xlim[0], xlim[1]],
        [xlim[0], xlim[1]],
        label=options.plot.scatter.oneone_line.label,
        c=options.plot.scatter.oneone_line.color,
        zorder=3,
    )

    if show_points is None or show_points:
        if show_density:
            c = z
            norm_ = norm
            cmap_ = cmap
        else:
            c = "0.25"
            norm_ = None
            cmap_ = None
        ax.scatter(
            x_sample,
            y_sample,
            c=c,
            s=options.plot.scatter.points.size,
            alpha=options.plot.scatter.points.alpha,
            marker=".",
            label=options.plot.scatter.points.label,
            zorder=1,
            norm=norm_,
            cmap=cmap_,
            **kwargs,
        )
    if len(xq) > 0:
        ax.plot(
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

    if reg_method:
        if fit_to_quantiles:
            slope, intercept = _linear_regression(
                obs=xq, model=yq, reg_method=reg_method
            )
        else:
            slope, intercept = _linear_regression(obs=x, model=y, reg_method=reg_method)

        ax.plot(
            x_trend,
            intercept + slope * x_trend,
            **settings.get_option("plot.scatter.reg_line.kwargs"),
            label=_reglabel(
                slope=slope, intercept=intercept, fit_to_quantiles=fit_to_quantiles
            ),
            zorder=2,
        )

    if show_hist:
        ax.hist2d(
            x,
            y,
            bins=nbins_hist,
            cmin=0.01,
            zorder=0.5,
            norm=norm,
            alpha=options.plot.scatter.points.alpha,
            cmap=cmap,
            **kwargs,
        )

    legend_kwargs = settings.get_option("plot.scatter.legend.kwargs")
    legend_kwargs["prop"] = {"size": options.plot.scatter.legend.fontsize}
    legend = ax.legend(**legend_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis("square")
    ax.set_xlim([xlim[0], xlim[1]])
    ax.set_ylim([ylim[0], ylim[1]])
    ax.minorticks_on()
    ax.grid(which="both", axis="both", linewidth="0.2", color="k", alpha=0.6)
    cbar = None
    # cmap = kwargs.get("cmap", None)
    if show_hist or (show_density and show_points):
        try:
            cbar = fig.colorbar(
                ScalarMappable(norm, cmap),
                ax=ax,
                fraction=0.046,
                pad=0.04,
                alpha=options.plot.scatter.points.alpha,
            )
            cbar.set_label("# points")
            cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        except ValueError:
            # too few points to make a colorbar
            pass

    cbar_width = _get_cbar_width(ax, cbar)

    ## Offset legend
    if cbar_width is not None:
        legend_loc = ax.transAxes.inverted().transform(
            legend.get_bbox_to_anchor().extents[0:2]
        )
        # If legend is outside the figure, move it to the right of the colorbar
        if legend_loc[0] > 1.0:
            legend.set_bbox_to_anchor((cbar_width + legend_loc[0], legend_loc[1]))

    # Add skill table
    if skill_scores is not None:
        _plot_summary_table(
            skill_scores,
            skill_score_unit,
            ax,
            cbar_width=cbar_width,
        )

    ax.set_title(title)

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
    reg_method,
    xlabel,
    ylabel,
    figsize,  # TODO not used by plotly, remove or keep for consistency?
    xlim,
    ylim,
    title,
    skill_scores,
    skill_score_unit,
    fit_to_quantiles,
    **kwargs,
):
    import plotly.graph_objects as go

    if "ax" in kwargs:
        ax = kwargs.pop("ax")
        if ax is not None:
            raise ValueError("Cannot pass matplotlib axes to plotly backend.")

    data = [
        go.Scatter(x=xlim, y=xlim, name="1:1", mode="lines", line=dict(color="blue")),
    ]

    if reg_method:
        if fit_to_quantiles:
            slope, intercept = _linear_regression(
                obs=xq, model=yq, reg_method=reg_method
            )
        else:
            slope, intercept = _linear_regression(obs=x, model=y, reg_method=reg_method)

        regression_line = go.Scatter(
            x=x_trend,
            y=intercept + slope * x_trend,
            name=_reglabel(
                slope=slope, intercept=intercept, fit_to_quantiles=fit_to_quantiles
            ),
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
    if len(xq) > 0:
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
    fig.update_xaxes(range=xlim, nticks=10)
    fig.update_yaxes(range=ylim, nticks=10)

    if skill_scores is not None:
        table = format_skill_table(skill_scores=skill_scores, unit=skill_score_unit)
        lines = [
            f"{row['name']:<6} {row['sep']} {row['value']:<6}"
            for _, row in table.iterrows()
        ]

        # add text box
        fig.add_annotation(
            x=0.99,
            y=0.01,
            xref="paper",
            yref="paper",
            text="<br>".join(lines),
            showarrow=False,
            align="left",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            font=dict(family="Consolas, 'Liberation Mono', monospace"),
        )

    fig.show()  # Should this be here


def _reglabel(slope: float, intercept: float, fit_to_quantiles: bool) -> str:
    sign = "" if intercept < 0 else "+"
    if fit_to_quantiles:
        fit = "QQ fit"
    else:
        fit = "Fit"
    return f"{fit}: y={slope:.2f}x{sign}{intercept:.2f}"


def _get_bins(bins: int | float, xymin, xymax) -> Tuple[int, float]:
    assert xymax >= xymin
    xyspan = xymax - xymin

    if isinstance(bins, int):
        nbins_hist: int = bins
        binsize: float = xyspan / nbins_hist
    elif isinstance(bins, float):
        binsize = bins
        nbins_hist = xyspan // binsize
    else:
        raise TypeError("bins must be a number")

    assert nbins_hist > 0

    return nbins_hist, binsize


def _plot_summary_border(
    figure_transform,
    x0,
    y0,
    dx,
    dy,
    borderpad=0.01,
    zorder=0,
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
        zorder=zorder,
        **bbox_kwargs,
    )

    plt.gca().add_patch(bbox)


def _get_cbar_width(ax, cbar=None) -> float | None:
    plt.draw()
    # If colorbar, get extents from colorbar label:
    if cbar is not None:
        label = cbar.ax.yaxis.get_label()
        if label is not None:
            x_extent_right = ax.transAxes.inverted().transform(
                label.get_window_extent()
            )[1][0]
        else:
            # If no label, get max value from colorbar ticks
            x_extent_right = cbar.bbox.transformed(ax.transAxes.inverted()).extents[2]
        return x_extent_right - 1
    else:
        return None


def _plot_summary_table(
    skill_scores: Mapping[str, float],
    units: str,
    ax,
    cbar_width: Optional[float] = None,
) -> None:
    # If colorbar, get extents from colorbar label:
    x0 = options.plot.scatter.skill_table.x_position
    if x0 > 1 and cbar_width is not None:
        x0 = cbar_width + x0

    # Plot table
    fontsize = options.plot.scatter.skill_table.fontsize
    ## Data
    table_data = format_skill_table(skill_scores, unit=units, sep="=")
    ## To get sizing, we plot a dummy table
    table_dummy = ax.table(
        table_data.values,
    )
    table_dummy.auto_set_font_size(False)
    table_dummy.set_fontsize(fontsize)

    col_widths = []
    renderer = ax.figure.canvas.get_renderer()
    for col_idx in range(table_data.values.shape[1]):  # Iterate over columns
        max_width = 0

        for row_idx in range(table_data.values.shape[0]):  # Iterate over rows
            cell = table_dummy[row_idx, col_idx]  # Get the cell object safely
            text = cell.get_text()
            bbox = text.get_window_extent(renderer, dpi=ax.figure.dpi)
            if col_idx != 1:  # Seoerator column
                padding = cell.PAD * ax.figure.dpi * 2
            else:
                padding = 0
            max_width = max(max_width, bbox.width + padding)
            height = bbox.height
        col_widths.append(max_width)

    # Remove dummy table
    table_dummy.remove()

    # Normalize widths
    ## These are the widths in axes coordinates
    widths_axTrans = [
        (
            ax.transAxes.inverted().transform((width, height))
            - ax.transAxes.inverted().transform((0, 0))
        )[0]
        for width in col_widths
    ]
    ## This is the height (assuming all the same)
    height_axTrans = (
        ax.transAxes.inverted().transform((0, height))
        - ax.transAxes.inverted().transform((0, 0))
    )[1]

    line_spacing = options.plot.scatter.skill_table.line_spacing
    line_padding = options.plot.scatter.skill_table.line_padding
    table_height = height_axTrans * line_spacing * table_data.values.shape[0]
    table_height_padded = table_height + line_padding * 2
    table1 = ax.table(
        table_data.values,
        loc="lower left",
        cellLoc="left",
        colWidths=col_widths,
        edges="open",
        bbox=[
            x0,
            1 - table_height - line_padding,
            np.sum(widths_axTrans),
            table_height,
        ],
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(fontsize)

    # Plot border
    _plot_summary_border(
        ax.transAxes,
        x0,
        1 - table_height_padded,
        np.sum(widths_axTrans),
        table_height_padded,
        borderpad=0,
        zorder=table1.get_zorder() - 1,
    )


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
