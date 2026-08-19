"""plotly renderers for the modelskill plots.

Every function here takes plain data (arrays, series, labels) and returns a
:class:`plotly.graph_objects.Figure`. They are the plotly counterparts of the
matplotlib code in the plotter classes and in `_scatter.py`, and are selected
by the ``backend="plotly"`` argument on the plot methods.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ..metrics import _linear_regression
from ..settings import options
from ._backend import (
    apply_layout,
    directional_axis,
    import_plotly_go,
    series_range,
)
from ._misc import format_skill_table, reglabel

# grey used for residual histograms, shared with the matplotlib backend
RESIDUAL_COLOR = "#8B8D8E"


def timeseries(
    *,
    obs: pd.Series,
    obs_color: str,
    mods: Mapping[str, pd.Series],
    mod_colors: Sequence[str],
    title: str | None = None,
    ylabel: str | None = None,
    ylim: Tuple[float, float] | None = None,
    figsize: Tuple[float, float] | None = None,
    directional: bool = False,
    **kwargs: Any,
):
    """Timeseries of observation and model data."""
    go = import_plotly_go()

    traces = [
        go.Scatter(
            x=mod.index,
            y=mod.values,
            name=name,
            line=dict(color=mod_colors[j]),
        )
        for j, (name, mod) in enumerate(mods.items())
    ]
    traces.append(
        go.Scatter(
            x=obs.index,
            y=obs.values,
            name=str(obs.name),
            mode="markers",
            marker=dict(color=obs_color),
        )
    )

    fig = go.Figure(traces)
    apply_layout(fig, figsize=figsize, title=title, yaxis_title=ylabel, **kwargs)
    if directional:
        directional_axis(fig, "y", ylim)
    else:
        fig.update_yaxes(range=ylim)
    return fig


def line(
    *,
    series: pd.Series,
    color: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    figsize: Tuple[float, float] | None = None,
    **kwargs: Any,
):
    """Line plot of a single time series."""
    go = import_plotly_go()

    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series.values,
            name=str(series.name),
            line=dict(color=color),
        )
    )
    apply_layout(fig, figsize=figsize, title=title, yaxis_title=ylabel, **kwargs)
    return fig


def histogram(
    *,
    series: Mapping[str, np.ndarray],
    colors: Sequence[str],
    bins: int | Sequence = 100,
    density: bool = True,
    alpha: float = 0.5,
    title: str | None = None,
    xlabel: str | None = None,
    figsize: Tuple[float, float] | None = None,
    directional: bool = False,
    **kwargs: Any,
):
    """Overlaid histograms of the given named data series."""
    go = import_plotly_go()

    nbins, bin_edges = _hist_bins(bins, series.values())

    traces = []
    for i, (name, values) in enumerate(series.items()):
        traces.append(
            go.Histogram(
                x=values,
                name=name,
                nbinsx=nbins,
                xbins=bin_edges,
                histnorm="probability density" if density else None,
                opacity=alpha,
                marker=dict(color=colors[i]),
            )
        )

    fig = go.Figure(traces)
    apply_layout(
        fig,
        figsize=figsize,
        title=title,
        xaxis_title=xlabel,
        yaxis_title="density" if density else "count",
        barmode="overlay",
        **kwargs,
    )
    if directional:
        directional_axis(fig, "x")
    return fig


def _hist_bins(bins: int | Sequence, series: Any) -> Tuple[int | None, Any]:
    """Translate a matplotlib `bins` argument to plotly nbinsx/xbins."""
    if isinstance(bins, (int, np.integer)):
        return int(bins), None
    edges = np.asarray(bins, dtype=float)
    if edges.size < 2:
        raise ValueError("`bins` must be an int or a sequence of at least two edges")
    return None, dict(start=edges[0], end=edges[-1], size=edges[1] - edges[0])


def kde(
    *,
    series: Mapping[str, np.ndarray],
    title: str | None = None,
    xlabel: str | None = None,
    figsize: Tuple[float, float] | None = None,
    directional: bool = False,
    bw_method: Any = None,
    n_points: int = 200,
    **kwargs: Any,
):
    """Kernel density estimates of the given named data series.

    The first series is drawn dashed, matching the matplotlib backend where
    the observation is dashed and the models are solid.
    """
    go = import_plotly_go()
    from scipy.stats import gaussian_kde

    xmin, xmax = series_range(list(series.values()))
    span = xmax - xmin
    grid = np.linspace(xmin - 0.1 * span, xmax + 0.1 * span, n_points)

    traces = []
    for i, (name, values) in enumerate(series.items()):
        density = gaussian_kde(np.asarray(values, dtype=float), bw_method=bw_method)
        traces.append(
            go.Scatter(
                x=grid,
                y=density(grid),
                name=name,
                mode="lines",
                line=dict(dash="dash" if i == 0 else "solid"),
            )
        )

    fig = go.Figure(traces)
    apply_layout(fig, figsize=figsize, title=title, xaxis_title=xlabel, **kwargs)
    # the density scale carries no information the user needs, as in matplotlib
    fig.update_yaxes(visible=False)
    if directional:
        directional_axis(fig, "x")
    return fig


def qq(
    *,
    quantiles: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: Tuple[float, float] | None = None,
    directional: bool = False,
    **kwargs: Any,
):
    """Quantile-quantile plot with a 1:1 line, one trace per model."""
    go = import_plotly_go()

    all_values = [v for pair in quantiles.values() for v in pair]
    xymin, xymax = series_range(all_values)

    traces = [
        go.Scatter(
            x=[xymin, xymax],
            y=[xymin, xymax],
            name=options.plot.scatter.oneone_line.label,
            mode="lines",
            line=dict(color=options.plot.scatter.oneone_line.color),
        )
    ]
    for name, (xq, yq) in quantiles.items():
        traces.append(
            go.Scatter(x=xq, y=yq, name=name, mode="lines+markers", marker=dict(size=4))
        )

    fig = go.Figure(traces)
    apply_layout(
        fig,
        figsize=figsize,
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        **kwargs,
    )
    if directional:
        directional_axis(fig, "x")
        directional_axis(fig, "y")
    else:
        fig.update_xaxes(range=(xymin, xymax))
        fig.update_yaxes(range=(xymin, xymax))
    return fig


def box(
    *,
    series: Mapping[str, np.ndarray],
    title: str | None = None,
    ylabel: str | None = None,
    figsize: Tuple[float, float] | None = None,
    directional: bool = False,
    **kwargs: Any,
):
    """Box plot with one box per named data series."""
    go = import_plotly_go()

    traces = [
        go.Box(y=np.asarray(values, dtype=float), name=name)
        for name, values in series.items()
    ]

    fig = go.Figure(traces)
    apply_layout(
        fig,
        figsize=figsize,
        title=title,
        yaxis_title=ylabel,
        showlegend=False,
        **kwargs,
    )
    if directional:
        directional_axis(fig, "y")
    return fig


def residual_hist(
    *,
    residuals: np.ndarray,
    bins: int | Sequence = 100,
    color: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    figsize: Tuple[float, float] | None = None,
    directional: bool = False,
    **kwargs: Any,
):
    """Histogram of model residuals."""
    go = import_plotly_go()

    nbins, bin_edges = _hist_bins(bins, [residuals])

    fig = go.Figure(
        go.Histogram(
            x=residuals,
            nbinsx=nbins,
            xbins=bin_edges,
            marker=dict(color=color or RESIDUAL_COLOR),
        )
    )
    apply_layout(
        fig,
        figsize=figsize,
        title=title,
        xaxis_title=xlabel,
        yaxis_title="count",
        showlegend=False,
        **kwargs,
    )
    if directional:
        fig.update_xaxes(
            tickmode="array", tickvals=np.linspace(-180, 180, 9), range=(-180, 180)
        )
    return fig


def scatter(
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
    norm,  # matplotlib-only, plotly scales its own colorbar
    show_hist,
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
    **kwargs,
):
    """Scatter plot of observation vs model, with 1:1 line and regression."""
    go = import_plotly_go()

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
            name=reglabel(
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

    fig = go.Figure(data=data)
    apply_layout(
        fig,
        figsize=figsize,
        legend=dict(x=0.01, y=0.99),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        title=dict(text=title, xanchor="center", yanchor="top", x=0.5, y=0.9),
        yaxis_title=ylabel,
        xaxis_title=xlabel,
        **kwargs,
    )
    fig.update_xaxes(range=xlim, nticks=10)
    fig.update_yaxes(range=ylim, nticks=10)

    if skill_scores is not None:
        _add_skill_table(fig, skill_scores=skill_scores, unit=skill_score_unit)

    return fig


def _add_skill_table(fig: Any, *, skill_scores: Mapping[str, float], unit: str) -> None:
    table = format_skill_table(skill_scores=skill_scores, unit=unit)
    lines = [
        f"{row['name']:<6} {row['sep']} {row['value']:<6}"
        for _, row in table.iterrows()
    ]
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


def taylor(
    *,
    points: Sequence[Any],
    obs_std: float,
    obs_text: str = "Observations",
    normalize_std: bool = False,
    title: str = "Taylor diagram",
    figsize: Tuple[float, float] | None = None,
    n_rms_contours: int = 5,
    **kwargs: Any,
):
    """Taylor diagram in a single-quadrant polar plot, r=std and theta=arccos(cc).

    Parameters
    ----------
    points : Sequence[TaylorPoint]
        the model points to show
    obs_std : float
        standard deviation of the observations (the reference radius)
    obs_text : str, optional
        label of the reference point, by default "Observations"
    normalize_std : bool, optional
        model std is normalized with observation std, by default False
    title : str, optional
        plot title, by default "Taylor diagram"
    figsize : (float, float), optional
        figure size in inches, by default None
    n_rms_contours : int, optional
        number of dotted centered-RMS-difference contours, by default 5
    **kwargs
        keyword arguments for fig.update_layout
    """
    go = import_plotly_go()

    stds = [p.std / p.obs_std if normalize_std else p.std for p in points]
    rmax = max([obs_std, *stds]) * 1.4

    traces = [
        _rms_contour(
            go, obs_std=obs_std, radius=r, name=f"RMSD={r:.2g}", showlegend=i == 0
        )
        for i, r in enumerate(_rms_contour_radii(obs_std, rmax, n_rms_contours))
    ]

    traces.append(
        go.Scatterpolar(
            r=[obs_std],
            theta=[0.0],
            name=obs_text,
            mode="markers",
            marker=dict(symbol="star", size=12, color="black"),
        )
    )
    for p, std in zip(points, stds):
        traces.append(
            go.Scatterpolar(
                r=[std],
                theta=[np.degrees(np.arccos(np.clip(p.cc, -1.0, 1.0)))],
                name=p.name,
                mode="markers",
                marker=dict(size=2 * p.marker_size),
                hovertemplate=f"{p.name}<br>std=%{{r:.3g}}<br>cc={p.cc:.3f}<extra></extra>",
            )
        )

    cc_ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
    fig = go.Figure(traces)
    fig.update_layout(
        polar=dict(
            sector=[0, 90],
            radialaxis=dict(
                range=[0, rmax],
                title=dict(
                    text="Std. dev." + (" (normalized)" if normalize_std else "")
                ),
                angle=45,
            ),
            angularaxis=dict(
                direction="counterclockwise",
                tickmode="array",
                tickvals=np.degrees(np.arccos(cc_ticks)),
                ticktext=[f"{c:g}" for c in cc_ticks],
            ),
        )
    )
    apply_layout(fig, figsize=figsize, title=title, **kwargs)
    return fig


def _rms_contour_radii(obs_std: float, rmax: float, n: int):
    """Radii of the centered-RMS-difference contours to draw"""
    step = rmax / (n + 1)
    return [step * (i + 1) for i in range(n)]


def _rms_contour(
    go: Any, *, obs_std: float, radius: float, name: str, showlegend: bool
):
    """A circle of constant centered RMS difference, in Taylor diagram polar coordinates"""
    t = np.linspace(0, 2 * np.pi, 180)
    x = obs_std + radius * np.cos(t)
    y = radius * np.sin(t)
    keep = y >= 0
    x, y = x[keep], y[keep]
    return go.Scatterpolar(
        r=np.hypot(x, y),
        theta=np.degrees(np.arctan2(y, x)),
        mode="lines",
        line=dict(color="lightgray", dash="dot", width=1),
        name=name,
        legendgroup="rmsd",
        showlegend=showlegend,
        hoverinfo="skip",
    )


def temporal_coverage(
    *,
    lines: Sequence[Tuple[str, Any, bool]],
    xlim: Tuple[Any, Any] | None = None,
    title: str | None = None,
    figsize: Tuple[float, float] | None = None,
    **kwargs: Any,
):
    """Temporal coverage of observations and models, one row per data source.

    Parameters
    ----------
    lines : Sequence of (name, times, is_model)
        the rows to draw; models are drawn as a line from first to last time,
        observations as markers at every time
    xlim : (datetime, datetime), optional
        limit the time axis, by default None
    title : str, optional
        plot title, by default None
    figsize : (float, float), optional
        figure size in inches, by default None
    **kwargs
        keyword arguments for fig.update_layout
    """
    go = import_plotly_go()

    traces = []
    for name, times, is_model in lines:
        if is_model:
            x, mode = [times[0], times[-1]], "lines"
        else:
            x, mode = list(times), "markers"
        traces.append(
            go.Scatter(
                x=x,
                y=[name] * len(x),
                name=name,
                mode=mode,
                marker=dict(symbol="line-ns", size=8, line=dict(width=1)),
            )
        )

    fig = go.Figure(traces)
    apply_layout(
        fig,
        figsize=figsize,
        title=title,
        showlegend=False,
        yaxis=dict(type="category"),
        **kwargs,
    )
    if xlim is not None:
        fig.update_xaxes(range=list(xlim))
    return fig


def spatial_overview(
    *,
    outlines: Sequence[np.ndarray],
    points: Sequence[Tuple[str, float, float]],
    tracks: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    title: str | None = None,
    figsize: Tuple[float, float] | None = None,
    **kwargs: Any,
):
    """Map of observation positions on the model domain outline.

    Parameters
    ----------
    outlines : Sequence of (n, 2) arrays
        model domain boundary polygons
    points : Sequence of (name, x, y)
        point observations, labelled on the map
    tracks : Sequence of (name, x, y)
        track observations
    title : str, optional
        plot title, by default "Spatial coverage"
    figsize : (float, float), optional
        figure size in inches, by default None
    **kwargs
        keyword arguments for fig.update_layout
    """
    go = import_plotly_go()

    traces = []
    for i, xy in enumerate(outlines):
        traces.append(
            go.Scatter(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="lines",
                line=dict(color="black", width=1),
                name="Domain",
                showlegend=i == 0,
                hoverinfo="skip",
            )
        )
    for name, x, y in tracks:
        traces.append(
            go.Scatter(x=x, y=y, mode="markers", name=name, marker=dict(size=3))
        )
    for name, px, py in points:
        traces.append(
            go.Scatter(
                x=[px],
                y=[py],
                mode="markers+text",
                name=name,
                text=[name],
                textposition="middle right",
                marker=dict(symbol="x", size=8),
            )
        )

    fig = go.Figure(traces)
    apply_layout(
        fig,
        figsize=figsize,
        title=title if title else "Spatial coverage",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        **kwargs,
    )
    return fig


def wind_rose(
    *,
    dir_centers: np.ndarray,
    dir_step: float,
    densities: Sequence[np.ndarray],
    mag_bins: np.ndarray,
    labels: Sequence[str],
    colorscales: Sequence[str],
    calm: float,
    calm_text: str = "Calm",
    rmax: float,
    r_ticks: np.ndarray,
    title: str | None = None,
    figsize: Tuple[float, float] | None = None,
    secondary_dir_step_factor: float = 2.0,
    legend: bool = True,
    **kwargs: Any,
):
    """Dual wind rose as stacked polar bars, with a calm hole in the centre.

    Parameters
    ----------
    dir_centers : np.ndarray
        centre of each directional sector, in degrees
    dir_step : float
        width of a directional sector, in degrees
    densities : Sequence of (n_mag, n_dir) arrays
        one array per dataset, fraction of data in each magnitude/direction bin
    mag_bins : np.ndarray
        magnitude bin edges, used for the legend labels; the last edge is an
        open-ended catch-all
    labels : Sequence[str]
        dataset names
    colorscales : Sequence[str]
        one plotly/matplotlib colorscale name per dataset
    calm : float
        radius of the calm hole
    calm_text : str, optional
        label of the calm hole, by default "Calm"
    rmax : float
        maximum radius beyond the calm hole
    r_ticks : np.ndarray
        radial tick positions (fractions), excluding the calm offset
    title : str, optional
        plot title, by default None
    figsize : (float, float), optional
        figure size in inches, by default None
    secondary_dir_step_factor : float, optional
        the secondary dataset is drawn with sectors this much narrower,
        by default 2.0
    legend : bool, optional
        show the magnitude legend, by default True
    **kwargs
        keyword arguments for fig.update_layout
    """
    go = import_plotly_go()

    traces = []
    n_mag = len(densities[0])
    for i, density in enumerate(densities):
        width = dir_step if i == 0 else dir_step / secondary_dir_step_factor
        colors = _sample_colorscale(colorscales[i], n_mag)
        for j in range(n_mag):
            # the last bin edge is an open-ended catch-all
            is_last = j == n_mag - 1
            name = (
                f">= {mag_bins[j]:.3g}"
                if is_last
                else f"{mag_bins[j]:.3g} - {mag_bins[j + 1]:.3g}"
            )
            traces.append(
                go.Barpolar(
                    r=density[j, :],
                    theta=dir_centers,
                    width=[width] * len(dir_centers),
                    name=name,
                    legendgroup=labels[i],
                    legendgrouptitle_text=labels[i] if j == 0 else None,
                    marker=dict(color=colors[j], line=dict(width=0)),
                    hovertemplate=(
                        f"{labels[i]}<br>{name}<br>"
                        "%{theta}&deg;<br>%{r:.1%}<extra></extra>"
                    ),
                )
            )

    fig = go.Figure(traces)
    fig.update_layout(
        barmode="stack",
        polar=dict(
            hole=calm / (calm + rmax) if (calm + rmax) > 0 else 0,
            radialaxis=dict(
                range=[0, rmax],
                tickmode="array",
                tickvals=r_ticks,
                ticktext=[f"{t * 100:.0f}%" for t in r_ticks],
                angle=5,
            ),
            angularaxis=dict(direction="clockwise", rotation=90),
        ),
    )
    if calm > 0:
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper", text=calm_text, showarrow=False
        )
    apply_layout(fig, figsize=figsize, title=title, showlegend=legend, **kwargs)
    return fig


def _sample_colorscale(cmap: str, n: int) -> list[str]:
    """n colors sampled from a matplotlib colormap, as plotly rgb strings"""
    import matplotlib as mpl

    colormap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    values = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.5])
    return [
        "rgb({:.0f},{:.0f},{:.0f})".format(*(np.array(colormap(v)[:3]) * 255))
        for v in values
    ]
