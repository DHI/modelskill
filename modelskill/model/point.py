from __future__ import annotations
from typing import Optional, Sequence, Any
import numpy as np

import xarray as xr
import pandas as pd

from ..obs import Observation, PointObservation
from ..types import PointType
from ..quantity import Quantity
from ..timeseries import TimeSeries, _parse_point_input
from ._base import Alignable


class PointModelResult(TimeSeries, Alignable):
    """Construct a PointModelResult from a 0d data source:
    dfs0 file, mikeio.Dataset/DataArray, pandas.DataFrame/Series
    or xarray.Dataset/DataArray

    Parameters
    ----------
    data : types.PointType
        the input data or file path
    name : Optional[str], optional
        The name of the model result,
        by default None (will be set to file name or item name)
    x : float, optional
        first coordinate of point position, by default None
    y : float, optional
        second coordinate of point position, by default None
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
        item: str | int | None = None,
        quantity: Optional[Quantity] = None,
        aux_items: Optional[Sequence[int | str]] = None,
    ) -> None:
        if not self._is_input_validated(data):
            data = _parse_point_input(
                data, name=name, item=item, quantity=quantity, aux_items=aux_items
            )

            data.coords["x"] = x
            data.coords["y"] = y
            data.coords["z"] = None  # TODO: or np.nan?

        assert isinstance(data, xr.Dataset)

        data_var = str(list(data.data_vars)[0])
        data[data_var].attrs["kind"] = "model"
        super().__init__(data=data)

    def extract(
        self, obs: PointObservation, spatial_method: Optional[str] = None
    ) -> PointModelResult:
        if not isinstance(obs, PointObservation):
            raise ValueError(f"obs must be a PointObservation not {type(obs)}")
        if spatial_method is not None:
            raise NotImplementedError(
                "spatial interpolation not possible when matching point model results with point observations"
            )
        return self

    def align(
        self,
        observation: Observation,
        *,
        max_gap: float | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Interpolate time series to new time index

        Parameters
        ----------
        new_time : pd.DatetimeIndex
            new time index
        **kwargs
            keyword arguments passed to xarray.interp()

        Returns
        -------
        xr.Dataset
            interpolated dataset
        """
        new_time = observation.time

        dati = self.data.dropna("time").interp(
            time=new_time, assume_sorted=True, **kwargs
        )

        pmr = PointModelResult(dati)
        if max_gap is not None:
            pmr = pmr._remove_model_gaps(mod_index=self.time, max_gap=max_gap)
        return pmr.data

    def _remove_model_gaps(
        self,
        mod_index: pd.DatetimeIndex,
        max_gap: float | None = None,
    ) -> PointModelResult:
        """Remove model gaps longer than max_gap from TimeSeries"""
        max_gap_delta = pd.Timedelta(max_gap, "s")
        valid_times = self._get_valid_times(mod_index, max_gap_delta)
        ds = self.data.sel(time=valid_times)
        return PointModelResult(ds)

    def _get_valid_times(
        self, mod_index: pd.DatetimeIndex, max_gap: pd.Timedelta
    ) -> pd.DatetimeIndex:
        """Used only by _remove_model_gaps"""
        obs_index = self.time
        # init dataframe of available timesteps and their index
        df = pd.DataFrame(index=mod_index)
        df["idx"] = range(len(df))

        # for query times get available left and right index of source times
        df = _interp_time(df, obs_index).dropna()
        df["idxa"] = np.floor(df.idx).astype(int)
        df["idxb"] = np.ceil(df.idx).astype(int)

        # time of left and right source times and time delta
        df["ta"] = mod_index[df.idxa]
        df["tb"] = mod_index[df.idxb]
        df["dt"] = df.tb - df.ta

        # valid query times where time delta is less than max_gap
        valid_idx = df.dt <= max_gap
        return df[valid_idx].index


def _interp_time(df: pd.DataFrame, new_time: pd.DatetimeIndex) -> pd.DataFrame:
    """Interpolate time series to new time index"""
    new_df = (
        df.reindex(df.index.union(new_time))
        .interpolate(method="time", limit_area="inside")
        .reindex(new_time)
    )
    return new_df
