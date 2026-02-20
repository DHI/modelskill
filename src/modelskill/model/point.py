from __future__ import annotations
from typing import Optional, Sequence, Any
import numpy as np

import xarray as xr
import pandas as pd

from ..obs import Observation
from ..types import PointType
from ..quantity import Quantity
from ..timeseries import TimeSeries, _parse_xyz_point_input


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


class PointModelResult(TimeSeries):
    """Model result for a single point location.

    Construct a PointModelResult from a 0d data source:
    dfs0 file, mikeio.Dataset/DataArray, pandas.DataFrame/Series
    or xarray.Dataset/DataArray

    Parameters
    ----------
    data : str, Path, mikeio.Dataset, mikeio.DataArray, pd.DataFrame, pd.Series, xr.Dataset or xr.DataArray
        filename (.dfs0 or .nc) or object with the data
    name : Optional[str], optional
        The name of the model result,
        by default None (will be set to file name or item name)
    x : float, optional
        first coordinate of point position, inferred from data if not given, else None
    y : float, optional
        second coordinate of point position, inferred from data if not given, else None
    z : float, optional
        third coordinate of point position, inferred from data if not given, else None
    item : str | int | None, optional
        If multiple items/arrays are present in the input an item
        must be given (as either an index or a string), by default None
    quantity : Quantity, optional
        Model quantity, for MIKE files this is inferred from the EUM information
    aux_items : Optional[list[int | str]], optional
        Auxiliary items, by default None
    """

    def __init__(
        self,
        data: PointType,
        *,
        name: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        item: str | int | None = None,
        quantity: Optional[Quantity] = None,
        aux_items: Optional[Sequence[int | str]] = None,
    ) -> None:
        if not self._is_input_validated(data):
            data = _parse_xyz_point_input(
                data,
                name=name,
                item=item,
                quantity=quantity,
                aux_items=aux_items,
                x=x,
                y=y,
                z=z,
            )

        assert isinstance(data, xr.Dataset)

        data_var = str(list(data.data_vars)[0])
        data[data_var].attrs["kind"] = "model"
        super().__init__(data=data)

    def interp_time(self, observation: Observation, **kwargs: Any) -> PointModelResult:
        """
        Interpolate model result to the time of the observation

        wrapper around xarray.Dataset.interp()

        Parameters
        ----------
        observation : Observation
            The observation to interpolate to
        **kwargs

            Additional keyword arguments passed to xarray.interp

        Returns
        -------
        PointModelResult
            Interpolated model result
        """
        ds = align_data(self.data, observation, **kwargs)
        return PointModelResult(ds)
