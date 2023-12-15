from __future__ import annotations
import warnings
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..metrics import metric_has_units, defined_metrics
from ..observation import unit_display_name


def _get_ax(ax=None, figsize=None):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _get_fig_ax(ax: plt.Axes | None = None, figsize=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.gcf()
    return fig, ax


def _xtick_directional(ax, xlim=None):
    """Set x-ticks for directional data"""
    ticks = ticks = _xyticks(lim=xlim)
    if len(ticks) > 2:
        ax.set_xticks(ticks)
    if xlim is None:
        ax.set_xlim(0, 360)


def _ytick_directional(ax, ylim=None):
    """Set y-ticks for directional data"""
    ticks = _xyticks(lim=ylim)
    if len(ticks) > 2:
        ax.set_yticks(ticks)
    if ylim is None:
        ax.set_ylim(0, 360)


def _xyticks(n_sectors=8, lim=None):
    """Set y-ticks for directional data"""
    # labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']
    ticks = np.linspace(0, 360, n_sectors + 1)
    if lim is not None:
        ticks = ticks[(ticks >= lim[0]) & (ticks <= lim[1])]
    return ticks


def sample_points(
    x: np.ndarray, y: np.ndarray, include: bool | int | float | None = None
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
    x: np.ndarray,
    y: np.ndarray,
    quantiles: Optional[Union[int, Sequence[float]]] = None,
):
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

    if isinstance(quantiles, Sequence):
        q = np.array(quantiles)
    else:
        if quantiles is None:
            if len(x) >= 3000:
                n_quantiles = 1000
            elif len(x) >= 300:
                n_quantiles = 100
            else:
                n_quantiles = 10

        if isinstance(quantiles, int):
            n_quantiles = quantiles

        q = np.linspace(0, 1, num=n_quantiles)

    return np.quantile(x, q=q), np.quantile(y, q=q)


def format_skill_df(df: pd.DataFrame, units: str, precision: int = 2):
    # select metrics columns
    accepted_columns = defined_metrics | {"n"}

    df = df.loc[:, df.columns.isin(accepted_columns)]

    # loop over series in dataframe, (columns)
    lines = [_format_skill_line(df[col], units, precision) for col in list(df.columns)]

    return np.array(lines)


def _format_skill_line(
    series: pd.Series,
    units: str,
    precision: int,
) -> Tuple[str, str, str]:
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
