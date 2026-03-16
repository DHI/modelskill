from __future__ import annotations
from typing import Any, Literal, Optional, Sequence

import numpy as np
import xarray as xr
import pandas as pd

from ..types import VerticalType
from ..obs import VerticalObservation
from ..quantity import Quantity
from ..timeseries import TimeSeries, _parse_vertical_input


class VerticalModelResult(TimeSeries):
    """Model result for a vertical column.

    Construct a VerticalColumnModelResult from a dfs0 file,
    mikeio.Dataset, pandas.DataFrame or a xarray.Datasets

    Parameters
    ----------
    data : types.ProfileType
        The input data or file path
    name : Optional[str], optional
        The name of the model result,
        by default None (will be set to file name or item name)
    item : str | int | None, optional
        If multiple items/arrays are present in the input an item
        must be given (as either an index or a string), by default None
    z_item : str | int | None, optional
        Item of the first coordinate of positions, by default None
    x : float, optional
        lateral coordinate of point position, inferred from data if not given, else None
    y : float, optional
        zonal coordinate of point position, inferred from data if not given, else None
    quantity : Quantity, optional
        Model quantity, for MIKE files this is inferred from the EUM information
    keep_duplicates : (str, bool), optional
        Strategy for handling duplicate timestamps (wraps xarray.Dataset.drop_duplicates)
        "first" to keep first occurrence, "last" to keep last occurrence,
        False to drop all duplicates, "offset" to add milliseconds to
        consecutive duplicates, by default "first"
    aux_items : Optional[list[int | str]], optional
        Auxiliary items, by default None
    """

    def __init__(
        self,
        data: VerticalType,
        *,
        name: Optional[str] = None,
        item: str | int | None = None,
        quantity: Optional[Quantity] = None,
        z_item: str | int = 0,
        x: Optional[float] = None,
        y: Optional[float] = None,
        keep_duplicates: Literal["first", "last", False] = "first",
        aux_items: Optional[Sequence[int | str]] = None,
    ) -> None:
        if not self._is_input_validated(data):
            data = _parse_vertical_input(
                data=data,
                name=name,
                item=item,
                quantity=quantity,
                z_item=z_item,
                x=x,
                y=y,
                keep_duplicates=keep_duplicates,
                aux_items=aux_items,
            )
        assert isinstance(data, xr.Dataset)
        data_var = str(list(data.data_vars)[0])
        data[data_var].attrs["kind"] = "model"
        super().__init__(data=data)

    # z coordinate not as property in TimeSeries. Add it here instead
    @property
    def z(self) -> Any:
        """z-coordinate"""
        return self._coordinate_values("z")

    # TODO: PHASE 3
    def _match_to_nearest_times(
        self, obs_df, mod_df, max_gap: pd.Timedelta | None = pd.Timedelta("30min")
    ) -> pd.DataFrame:
        obs_times = obs_df.index.unique().sort_values()
        mod_times_unique = mod_df.index.unique().sort_values()

        # get_indexer requires a unique, monotonic index - work on unique times first
        idx = mod_times_unique.get_indexer(
            obs_times, method="nearest", tolerance=max_gap
        )
        valid = idx != -1

        matched_mod_times = mod_times_unique[idx[valid]]
        obs_times_valid = obs_times[valid]

        return pd.DataFrame(
            {"obs_time": obs_times_valid, "mod_time": matched_mod_times}
        )

    def _interpolate_to_obs_depths(
        self,
        obs_df,
        mod_df,
        obs_times_valid,
        matched_mod_times,
        *,
        obs_value_col: str,
        mod_value_col: str,
    ) -> pd.DataFrame:
        records = []

        for obs_t, mod_t in zip(obs_times_valid, matched_mod_times):
            obs_at_t = obs_df.loc[[obs_t]].sort_values("z")
            mod_at_t = mod_df.loc[[mod_t]].sort_values("z")

            obs_z = obs_at_t["z"].to_numpy(dtype=float)
            obs_values = obs_at_t[obs_value_col].to_numpy(dtype=float)
            mod_z = mod_at_t["z"].to_numpy(dtype=float)
            mod_values = mod_at_t[mod_value_col].to_numpy(dtype=float)

            if mod_z.size < 2:
                continue

            valid_mod = np.isfinite(mod_z) & np.isfinite(mod_values)
            if not np.any(valid_mod):
                continue
            mod_z = mod_z[valid_mod]
            mod_values = mod_values[valid_mod]

            # np.interp requires monotonic x-values and behaves poorly on duplicates.
            # Keep first occurrence for duplicate depths.
            _, unique_idx = np.unique(mod_z, return_index=True)
            unique_idx = np.sort(unique_idx)
            mod_z = mod_z[unique_idx]
            mod_values = mod_values[unique_idx]

            if mod_z.size < 2:
                continue

            z_lo, z_hi = mod_z.min(), mod_z.max()
            in_range = (
                np.isfinite(obs_z)
                & np.isfinite(obs_values)
                & (obs_z >= z_lo)
                & (obs_z <= z_hi)
            )

            if not np.any(in_range):
                continue

            mod_interp = np.interp(obs_z[in_range], mod_z, mod_values)
            for z, obs_v, mod_v in zip(
                obs_z[in_range], obs_values[in_range], mod_interp
            ):
                records.append({"time": obs_t, "z": z, "obs": obs_v, "mod": mod_v})

        if not records:
            return pd.DataFrame(
                columns=["z", "obs", "mod"], index=pd.Index([], name="time")
            )

        return pd.DataFrame(records).set_index("time")

    def align_to_obs_profiles(self, observation: VerticalObservation) -> xr.Dataset:
        # In future, support contained in z as well. For now, just vertical interpolation
        _match_to_nearest_times = self._match_to_nearest_times(
            observation.data[["z"]].to_dataframe(), self.data[["z"]].to_dataframe()
        )

        # Interpolate model to obs depths at matched times
        pairs = self._interpolate_to_obs_depths(
            observation.data.to_dataframe(),
            self.data.to_dataframe(),
            _match_to_nearest_times["obs_time"],
            _match_to_nearest_times["mod_time"],
            obs_value_col=observation.name,
            mod_value_col=self.name,
        )

        return pairs.reset_index().set_index(["time", "z"]).to_xarray()
