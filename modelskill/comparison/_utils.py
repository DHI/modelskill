from __future__ import annotations
from typing import Optional, Iterable, Callable, List, Union
from datetime import datetime
import numpy as np
import pandas as pd

from .. import metrics as mtr

TimeTypes = Union[str, np.datetime64, pd.Timestamp, datetime]
IdOrNameTypes = Union[int, str, List[int], List[str]]


def _parse_metric(metric, default_metrics, return_list=False):
    if metric is None:
        metric = default_metrics

    if isinstance(metric, (str, Callable)):
        metric = mtr.get_metric(metric)
    elif isinstance(metric, Iterable):
        metrics = [_parse_metric(m, default_metrics) for m in metric]
        return metrics
    elif not callable(metric):
        raise TypeError(f"Invalid metric: {metric}. Must be either string or callable.")
    if return_list:
        if callable(metric) or isinstance(metric, str):
            metric = [metric]
    return metric


def _add_spatial_grid_to_df(
    df: pd.DataFrame, bins, binsize: Optional[float]
) -> pd.DataFrame:
    if binsize is None:
        # bins from bins
        if isinstance(bins, tuple):
            bins_x = bins[0]
            bins_y = bins[1]
        else:
            bins_x = bins
            bins_y = bins
    else:
        # bins from binsize
        x_ptp = df.x.values.ptp()  # type: ignore
        y_ptp = df.y.values.ptp()  # type: ignore
        nx = int(np.ceil(x_ptp / binsize))
        ny = int(np.ceil(y_ptp / binsize))
        x_mean = np.round(df.x.mean())
        y_mean = np.round(df.y.mean())
        bins_x = np.arange(
            x_mean - nx / 2 * binsize, x_mean + (nx / 2 + 1) * binsize, binsize
        )
        bins_y = np.arange(
            y_mean - ny / 2 * binsize, y_mean + (ny / 2 + 1) * binsize, binsize
        )
    # cut and get bin centre
    df["xBin"] = pd.cut(df.x, bins=bins_x)
    df["xBin"] = df["xBin"].apply(lambda x: x.mid)
    df["yBin"] = pd.cut(df.y, bins=bins_y)
    df["yBin"] = df["yBin"].apply(lambda x: x.mid)

    return df


def _groupby_df(df, by, metrics, n_min: Optional[int] = None):
    def calc_metrics(x):
        row = {}
        row["n"] = len(x)
        for metric in metrics:
            row[metric.__name__] = metric(x.obs_val, x.mod_val)
        return pd.Series(row)

    # .drop(columns=["x", "y"])

    res = df.groupby(by=by, observed=False).apply(calc_metrics)

    if n_min:
        # nan for all cols but n
        cols = [col for col in res.columns if not col == "n"]
        res.loc[res.n < n_min, cols] = np.nan

    res["n"] = res["n"].fillna(0)
    res = res.astype({"n": int})

    return res


def _parse_groupby(by, n_models, n_obs, n_var=1):
    if by is None:
        by = []
        if n_models > 1:
            by.append("model")
        if n_obs > 1:  # or ((n_models == 1) and (n_obs == 1)):
            by.append("observation")
        if n_var > 1:
            by.append("variable")
        if len(by) == 0:
            # default value
            by.append("observation")
        return by

    if isinstance(by, str):
        if by in {"mdl", "mod", "models"}:
            by = "model"
        if by in {"obs", "observations"}:
            by = "observation"
        if by in {"var", "variables", "item"}:
            by = "variable"
        if by[:5] == "freq:":
            freq = by.split(":")[1]
            by = pd.Grouper(freq=freq)
    elif isinstance(by, Iterable):
        by = [_parse_groupby(b, n_models, n_obs, n_var) for b in by]
        return by
    else:
        raise ValueError("Invalid by argument. Must be string or list of strings.")
    return by
