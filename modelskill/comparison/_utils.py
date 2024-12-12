from __future__ import annotations
from typing import Callable, Optional, Iterable, List, Sequence, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd
import polars as pl

from modelskill.metrics import get_metric


TimeTypes = Union[str, np.datetime64, pd.Timestamp, datetime]
IdxOrNameTypes = Union[int, str, List[int], List[str]]


def _add_spatial_grid_to_df(
    df: pd.DataFrame, bins, binsize: Optional[float]
) -> pd.DataFrame:
    if isinstance(df, pl.DataFrame):
        # convert to pandas
        df = df.to_pandas()  # .set_index(["x", "y"])

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
        x_ptp = np.ptp(df.x.values)  # type: ignore
        y_ptp = np.ptp(df.y.values)  # type: ignore
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

    # TODO avoid using pandas
    return pl.DataFrame(df.to_dict(orient="records"))


def _groupby_df(
    df: pl.DataFrame,
    *,
    by: Sequence[str],
    metrics: Sequence[str | Callable],
    n_min: Optional[int] = None,
) -> pl.DataFrame:
    if _dt_in_by(by):
        df, by = _add_dt_to_df(df, by)

    # TODO extract method
    obs = pl.col("obs_val")
    mod = pl.col("mod_val")
    residual = mod - obs
    uresidual = residual - residual.mean()
    r = pl.corr("obs_val", "mod_val")

    NAMED_METRICS = {
        "bias": residual.mean().alias("bias"),
        "rmse": residual.pow(2).mean().sqrt().alias("rmse"),
        "urmse": uresidual.pow(2).mean().sqrt().alias("urmse"),
        "mae": residual.abs().mean().alias("mae"),
        "r2": (1 - residual.pow(2).sum() / obs.sub(obs.mean()).pow(2).sum()).alias(
            "r2"
        ),
        "nse": (1 - residual.pow(2).sum() / obs.sub(obs.mean()).pow(2).sum()).alias(
            "nse"
        ),
        "cc": pl.corr("obs_val", "mod_val").alias("cc"),
        "n": pl.col("obs_val").count().alias("n"),
        "si": (uresidual.pow(2).mean().sqrt() / obs.abs().mean()).alias("si"),
        "max_error": residual.abs().max().alias("max_error"),
        "kge": (
            1
            - (
                (r - 1).pow(2)
                + (mod.std() / obs.std() - 1.0).pow(2)
                + (mod.mean() / obs.mean() - 1.0).pow(2).sqrt()
            )
        ).alias("kge"),
        "_std_obs": pl.col("obs_val").std().alias("_std_obs"),
        "_std_mod": pl.col("mod_val").std().alias("_std_mod"),
        "lin_slope": (pl.cov("obs_val", "mod_val") / pl.col("obs_val").var()).alias(
            "lin_slope"
        ),
    }

    sel_metrics = [
        NAMED_METRICS[str(metric)] for metric in metrics if metric in NAMED_METRICS
    ]

    temporal_aggregation = False
    for group in by:
        if "freq:" in group:
            temporal_aggregation = True
            _, every = group.split(":")

    if temporal_aggregation:
        by = [b for b in by if "freq:" not in b]
        res = (
            df.sort("time")
            .group_by_dynamic("time", every=every, group_by=by)
            .agg(*sel_metrics)
        )
    else:
        res = df.group_by(by).agg(*sel_metrics)

    def _get_metric(metric: Callable | str) -> Callable:
        if callable(metric):
            return metric
        return get_metric(metric)

    # This is a bit of a hack to allow for peak_ratio which is special, since it uses datetime information
    PR = False
    if "peak_ratio" in metrics:
        metrics = [m for m in metrics if m != "peak_ratio"]
        PR = True
    custom_metrics = [_get_metric(m) for m in metrics if m not in NAMED_METRICS]

    if len(custom_metrics) > 0:
        cres = _calculate_custom_metrics(df, by, custom_metrics)
        res = res.join(cres, on=by)

    if PR:
        pres = _calculate_peak_ratio(df, by)
        res = res.join(pres, on=by)

    if n_min:
        # TODO why not simply keep rows where n > n_min?
        # res = res.filter(pl.col("n") > n_min)

        # This seems convoluted

        non_n_metrics = [m for m in metrics if m != "n"]

        res = res.select(
            pl.col("x"),
            pl.col("y"),
            pl.col("n"),
            pl.when(pl.col("n") >= n_min).then(pl.col(*non_n_metrics)),  # type: ignore
        )

    return res


def _calculate_custom_metrics(
    df: pl.DataFrame, by: Sequence[str], custom_metrics: Sequence[Callable]
) -> pl.DataFrame:
    cres = df.group_by(by).agg(
        [
            pl.struct(["obs_val", "mod_val"])
            .map_elements(
                lambda combined, metric=metric: metric(  # type: ignore
                    # TODO this doesn't work with peak_ratio which expects a pd.Series with a DateTimeIndex
                    combined.struct.field("obs_val").to_numpy(),
                    combined.struct.field("mod_val").to_numpy(),
                )
            )
            .alias(metric.__name__)
            for metric in custom_metrics
        ]
    )
    return cres


def _calculate_peak_ratio(df: pl.DataFrame, by: Sequence[str]) -> pl.DataFrame:
    from modelskill.metrics import peak_ratio

    # peak ratio expects a pd.Series with a DateTimeIndex
    # TODO reconsider this approach
    expr = (
        pl.struct(["time", "obs_val", "mod_val"])
        .map_elements(
            lambda combined: peak_ratio(  # type: ignore
                pd.Series(
                    combined.struct.field("obs_val").to_numpy(),
                    index=pd.DatetimeIndex(combined.struct.field("time").to_pandas()),
                ),
                pd.Series(
                    combined.struct.field("mod_val").to_numpy(),
                    index=pd.DatetimeIndex(combined.struct.field("time").to_pandas()),
                ),
            )
        )
        .alias("peak_ratio")
    )
    pres = df.group_by(by).agg(expr)
    return pres


def _dt_in_by(by):
    by = [by] if isinstance(by, str) else by
    if any(str(by).startswith("dt:") for by in by):
        return True
    return False


EXPRS = {
    "year": pl.col("time").dt.year(),
    "quarter": pl.col("time").dt.quarter(),
    "month": pl.col("time").dt.month(),
    # "month_name", # TODO available in polars-xdt
    "day": pl.col("time").dt.day(),
    "day_of_week": pl.col("time").dt.weekday(),
    "dayofweek": pl.col("time").dt.weekday(),
    "day_of_year": pl.col("time").dt.ordinal_day(),
    "hour": pl.col("time").dt.hour(),
    "minute": pl.col("time").dt.minute(),
    "second": pl.col("time").dt.second(),
    "weekday": pl.col("time").dt.weekday(),
}

ALLOWED_DT = list(EXPRS.keys())


def _add_dt_to_df(
    df: pl.DataFrame, by: Sequence[str]
) -> Tuple[pl.DataFrame, List[str]]:
    assert isinstance(by, list)

    for j, b in enumerate(by):
        assert isinstance(b, str)
        if str(b).startswith("dt:"):
            dt_str = b.split(":")[1].lower()
            if dt_str not in ALLOWED_DT:
                raise ValueError(
                    f"Invalid Pandas dt accessor: {dt_str}. Allowed values are: {ALLOWED_DT}"
                )

            if dt_str in df.columns:
                raise ValueError(
                    f"Cannot use datetime attribute {dt_str} as it already exists in the dataframe."
                )
            # df[dt_str] = ser
            # by[j] = dt_str  # remove 'dt:' prefix
            df = df.with_columns(EXPRS[dt_str].alias(dt_str))
            by[j] = dt_str
    # by = by[0] if len(by) == 1 else by
    return df, by


def _parse_groupby(
    by: str | Iterable[str] | None, *, n_mod: int, n_qnt: int
) -> List[str | pd.Grouper]:
    if by is None:
        cols: List[str | pd.Grouper]
        cols = ["model", "observation", "quantity"]

        if n_mod == 1:
            cols.remove("model")
        if n_qnt == 1:
            cols.remove("quantity")
        return cols

    if isinstance(by, str):
        cols = [by]
    elif isinstance(by, Iterable):
        cols = list(by)

    res = []
    for col in cols:
        # if col[:5] == "freq:":
        #    freq = col.split(":")[1]
        #    res.append(pd.Grouper(key="time", freq=freq))
        # else:
        res.append(col)

    ress = set(res)
    # TODO it seems like observation should not always be added
    if len(ress) == 0:
        ress.add("observation")
    # ress.add("observation")
    return list(ress)
