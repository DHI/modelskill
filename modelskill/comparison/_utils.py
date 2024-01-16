from __future__ import annotations
from typing import Callable, Optional, Iterable, List, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd


TimeTypes = Union[str, np.datetime64, pd.Timestamp, datetime]
IdxOrNameTypes = Union[int, str, List[int], List[str]]


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


def _groupby_df(
    df: pd.DataFrame,
    by: List[str],
    metrics: List[Callable],
    n_min: Optional[int] = None,
) -> pd.DataFrame:
    def calc_metrics(group):
        row = {}
        row["n"] = len(group)
        for metric in metrics:
            row[metric.__name__] = metric(group.obs_val, group.mod_val)
        return pd.Series(row)

    if _dt_in_by(by):
        df, by = _add_dt_to_df(df, by)

    # sort=False to avoid re-ordering compared to original cc (also for performance)
    res = df.groupby(by=by, observed=False, sort=False).apply(calc_metrics)

    if n_min:
        # nan for all cols but n
        cols = [col for col in res.columns if not col == "n"]
        res.loc[res.n < n_min, cols] = np.nan

    res["n"] = res["n"].fillna(0)
    res = res.astype({"n": int})

    return res


def _dt_in_by(by):
    by = [by] if isinstance(by, str) else by
    if any(str(by).startswith("dt:") for by in by):
        return True
    return False


ALLOWED_DT = [
    "year",
    "quarter",
    "month",
    "month_name",
    "day",
    "day_of_year",
    "dayofyear",
    "day_of_week",
    "dayofweek",
    "hour",
    "minute",
    "second",
    "weekday",
]


def _add_dt_to_df(df: pd.DataFrame, by: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    ser = df["time"]
    assert isinstance(by, list)
    # by = [by] if isinstance(by, str) else by

    for j, b in enumerate(by):
        assert isinstance(b, str)
        if str(b).startswith("dt:"):
            dt_str = b.split(":")[1].lower()
            if dt_str not in ALLOWED_DT:
                raise ValueError(
                    f"Invalid Pandas dt accessor: {dt_str}. Allowed values are: {ALLOWED_DT}"
                )
            ser = ser.dt.__getattribute__(dt_str)
            if dt_str in df.columns:
                raise ValueError(
                    f"Cannot use datetime attribute {dt_str} as it already exists in the dataframe."
                )
            df[dt_str] = ser
            by[j] = dt_str  # remove 'dt:' prefix
    # by = by[0] if len(by) == 1 else by
    return df, by


def _parse_groupby(
    by: Iterable[str] | None, n_models: int, n_obs: int, n_qnt: int = 1
) -> List[str | pd.core.resample.TimeGrouper]:
    if by is None:
        by = []
        if n_models > 1:
            by.append("model")
        if n_obs > 1:  # or ((n_models == 1) and (n_obs == 1)):
            by.append("observation")
        if n_qnt > 1:
            by.append("quantity")
        if len(by) == 0:
            # default value
            by.append("observation")
        return by

    if isinstance(by, str):
        if by in {"mdl", "mod", "models"}:
            by = "model"
        if by in {"obs", "observations"}:
            by = "observation"
        if by in {"var", "quantities", "item"}:
            by = "quantity"
        if by[:5] == "freq:":
            freq = by.split(":")[1]
            by = pd.Grouper(key="time", freq=freq)
        return [by]
    elif isinstance(by, Iterable):
        by = [_parse_groupby(b, n_models, n_obs, n_qnt)[0] for b in by]
        return by
    else:
        raise ValueError("Invalid by argument. Must be string or list of strings.")
    return by
