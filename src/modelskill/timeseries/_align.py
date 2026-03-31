"""Time series alignment utilities."""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
from typing import Any
from ..obs import Observation


def _get_valid_times(
    obs_time: pd.DatetimeIndex, mod_index: pd.DatetimeIndex, max_gap: pd.Timedelta
) -> pd.DatetimeIndex:
    """Get valid times where interpolation gaps are within max_gap"""
    # init dataframe of available timesteps and their index
    df = pd.DataFrame(index=mod_index)
    df["idx"] = range(len(df))

    # for query times get available left and right index of source times
    df = (
        df.reindex(df.index.union(obs_time))
        .interpolate(method="time", limit_area="inside")
        .reindex(obs_time)
        .dropna()
    )
    df["idxa"] = np.floor(df.idx).astype(int)
    df["idxb"] = np.ceil(df.idx).astype(int)

    # time of left and right source times and time delta
    df["ta"] = mod_index[df.idxa]
    df["tb"] = mod_index[df.idxb]
    df["dt"] = df.tb - df.ta

    # valid query times where time delta is less than max_gap
    valid_idx = df.dt <= max_gap
    return df[valid_idx].index


def _remove_model_gaps(
    data: xr.Dataset,
    mod_index: pd.DatetimeIndex,
    max_gap: float | None = None,
) -> xr.Dataset:
    """Remove model gaps longer than max_gap from Dataset"""
    max_gap_delta = pd.Timedelta(max_gap, "s")
    obs_time = pd.DatetimeIndex(data.time.to_index())
    valid_times = _get_valid_times(obs_time, mod_index, max_gap_delta)
    return data.sel(time=valid_times)


def align_data(
    data: xr.Dataset,
    observation: Observation,
    *,
    max_gap: float | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Align model data to observation time.

    Interpolates model result to the time of the observation.

    Parameters
    ----------
    data : xr.Dataset
        The model dataset to align
    observation : Observation
        The observation to align to
    max_gap : float | None, optional
        Maximum gap in seconds for interpolation gaps removal, by default None
    **kwargs : Any
        Additional keyword arguments passed to xarray.interp

    Returns
    -------
    xr.Dataset
        Aligned dataset
    """
    new_time = observation.time

    dati = data.dropna("time").interp(time=new_time, assume_sorted=True, **kwargs)

    if max_gap is not None:
        model_time = pd.DatetimeIndex(data.time.to_index())
        dati = _remove_model_gaps(dati, mod_index=model_time, max_gap=max_gap)
    return dati
