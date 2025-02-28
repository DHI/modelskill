from __future__ import annotations
import warnings
from typing import Optional, Sequence, Tuple, Union, Mapping

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from ..metrics import metric_has_units, defined_metrics
from ..obs import unit_display_name


def _get_ax(ax=None, figsize=None):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _get_fig_ax(ax: Axes | None = None, figsize=None):
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
    x: np.ndarray, y: np.ndarray, show_points: bool | int | float | None = None
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

    if show_points is True:
        return x, y

    if show_points is None:
        if len(x) < 5e4:
            return x, y
        else:
            show_points = 50000
            warnings.warn(
                message=f"Showing only {show_points} points in plot. Set `show_points` to True to show all points."
            )
    else:
        if not isinstance(show_points, (bool, int, float)):
            raise TypeError(
                f"'show_points' must be bool, int or float, not {type(show_points)}"
            )

    if show_points is False:
        return np.array([]), np.array([])

    if isinstance(show_points, float):
        if not 0 <= show_points <= 1:
            raise ValueError("`show_points` fraction must be in [0,1]")

        n_samples = int(len(x) * show_points)
    elif isinstance(show_points, int):
        if show_points < 0:
            raise ValueError("`show_points` must be positive integer")
        if show_points > len(x):
            show_points = len(x)
        n_samples = show_points

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


def format_skill_table(
    skill_scores: Mapping[str, float],
    unit: str,
    sep: str = " =  ",
    skill_score_names: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    # select metrics columns
    accepted_columns = defined_metrics | {"n"}
    kv = {k: v for k, v in skill_scores.items() if k in accepted_columns}
    if skill_score_names is None:
        skill_score_names = {}
    mapped_names = {
        k: skill_score_names[k] if k in skill_score_names else None for k in kv.keys()
    }

    lines = [
        _format_skill_line(key, value, unit, sep=sep, mapped_name=mapped_names[key])
        for key, value in kv.items()
    ]

    # TODO add sign and unit columns
    df = pd.DataFrame(lines, columns=["name", "sep", "value"])
    return df


def _format_skill_line(
    metric_name: str,
    value: float | int | str,
    unit: str,
    sep: str = " =  ",
    mapped_name: Optional[str] = None,
) -> Tuple[str, str, str]:
    precision: int = 2
    item_unit = " "
    fvalue = str(value)

    if metric_name == "n":
        fvalue = str(int(value))
    else:
        if metric_has_units(metric=metric_name):
            # if statistic has dimensions, then add units
            item_unit = unit_display_name(unit)

        if isinstance(value, (float, int)):
            rounded_value = np.round(value, precision)
            fmt = f".{precision}f"
            fvalue = f"{rounded_value:{fmt}}"
        else:
            fvalue = str(value)
    if mapped_name:
        name = mapped_name
    else:
        name = metric_name.upper()

    return f"{name}", sep, f"{fvalue} {item_unit}"
