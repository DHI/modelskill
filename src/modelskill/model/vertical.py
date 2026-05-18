from __future__ import annotations
from typing import Any, Literal, Sequence

import xarray as xr
import pandas as pd
import numpy as np

from ..types import VerticalType
from ..quantity import Quantity
from ..timeseries import TimeSeries, _parse_vertical_input
from ..obs import VerticalObservation


class VerticalModelResult(TimeSeries):
    """Model result for a vertical profile at a fixed (x, y) location.

    The input must be in long format: one row per (time, z) pair, with a
    column/item for the vertical coordinate and a column/item for the
    modelled value. At least two items are required (z + value); if more are
    present, ``item`` must be given.

    Parameters
    ----------
    data : str, Path, pd.DataFrame, mikeio.Dfs0, mikeio.Dataset, xr.Dataset
        Input data or path to a dfs0 file.
    name : str, optional
        Name of the model result, by default the file or item name.
    item : str or int, optional
        Index or name of the value item. Required if the input has more than
        two items.
    z_item : str or int, optional
        Index or name of the item holding the vertical coordinate, by default 0.
    x : float, optional
        x-coordinate of the profile location, inferred from data when possible.
    y : float, optional
        y-coordinate of the profile location, inferred from data when possible.
    quantity : Quantity, optional
        Model quantity. For MIKE files this is inferred from EUM information.
    keep_duplicates : {"first", "last", False}, optional
        Strategy for handling duplicate (time, z) pairs, by default "first".
    aux_items : list[int | str], optional
        Auxiliary items to keep alongside the value item.

    Notes
    -----
    The input must be in long format: one row per (time, z) pair, with one
    item/column holding the vertical coordinate and another holding the
    modelled value. A dfs0 with N depth levels has its profile timestamps
    repeated N times on a non-equidistant time axis.

    Examples
    --------
    From a `pandas.DataFrame` in long format:

    ```{python}
    import modelskill as ms
    import pandas as pd

    times = pd.to_datetime(
        ["2010-01-01 01:00"] * 3 + ["2010-01-01 02:00"] * 3
    )
    df = pd.DataFrame(
        {"z": [0.0, -5.0, -10.0, 0.0, -5.0, -10.0],
         "Salinity": [30.1, 30.3, 30.4, 30.5, 30.3, 30.3]},
        index=times,
    )
    ms.VerticalModelResult(df, item="Salinity", z_item="z", x=12.0, y=55.0)
    ```

    From a dfs0 file (with z, Salinity and Temperature items):

    ```{python}
    ms.VerticalModelResult(
        "../data/vertical/VerticalModel_at_obs.dfs0",
        item="Salinity",
        z_item="z",
        x=12.0,
        y=55.0,
    )
    ```
    """

    def __init__(
        self,
        data: VerticalType,
        *,
        name: str | None = None,
        item: str | int | None = None,
        quantity: Quantity | None = None,
        z_item: str | int = 0,
        x: float | None = None,
        y: float | None = None,
        keep_duplicates: Literal["first", "last", False] = "first",
        aux_items: Sequence[int | str] | None = None,
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

    @property
    def z(self) -> Any:
        """z-coordinate"""
        return self._coordinate_values("z")

    def _match_to_nearest_times(
        self, obs_df: pd.DataFrame, t_tol: pd.Timedelta | None = None
    ) -> pd.DataFrame:
        """Match model times to nearest observation times within a specified tolerance."""
        mod_df = self.data[["z"]].to_dataframe()
        obs_times = obs_df.index.unique().sort_values()
        mod_times_unique = mod_df.index.unique().sort_values()

        # get_indexer requires a unique, monotonic index - work on unique times first
        idx = mod_times_unique.get_indexer(obs_times, method="nearest", tolerance=t_tol)
        valid = idx != -1

        matched_mod_times = mod_times_unique[idx[valid]]
        obs_times_valid = obs_times[valid]

        return pd.DataFrame(
            {"obs_time": obs_times_valid, "mod_time": matched_mod_times}
        )

    def _interpolate_to_obs_depths(
        self,
        obs_df: pd.DataFrame,
        obs_times_valid: pd.Series,
        matched_mod_times: pd.Series,
        *,
        mod_value_col: str,
    ) -> pd.DataFrame:
        """Interpolate model values to observation depths for matched times."""
        records = []

        mod_df = self.data.to_dataframe()

        for obs_t, mod_t in zip(obs_times_valid, matched_mod_times):
            obs_at_t = obs_df.loc[[obs_t]]
            mod_at_t = mod_df.loc[[mod_t]]

            obs_z = obs_at_t["z"].to_numpy(dtype=float)

            # Sort model depths for np.interp
            mod_at_t_sorted = mod_at_t.sort_values("z")
            m_z = mod_at_t_sorted["z"].to_numpy(dtype=float)
            m_v = mod_at_t_sorted[mod_value_col].to_numpy(dtype=float)

            # make sure we have at least 2 points to interpolate, otherwise skip this time step
            if m_z.size < 2:
                continue

            mod_interp = np.interp(obs_z, m_z, m_v, left=np.nan, right=np.nan)
            for z, mod_v in zip(obs_z, mod_interp):
                records.append({"time": obs_t, "z": z, self.name: mod_v})

        if not records:
            return pd.DataFrame(
                columns=["z", self.name], index=pd.Index([], name="time")
            )

        return pd.DataFrame(records).set_index("time")

    def align(
        self, vo: VerticalObservation, temporal_tolerance: pd.Timedelta | None = None
    ) -> xr.Dataset:
        """Align model result to observation by matching nearest times and interpolating to observation depths.

        Observation depths outside the model depth range are assigned NaN values; no extrapolation is performed.

        Parameters
        ----------
        vo : VerticalObservation
            Vertical observation to align with
        temporal_tolerance : pd.Timedelta, optional
            Maximum allowed time difference for matching, by default None

        Returns
        -------
        xr.Dataset
            Aligned model result

        """
        # if temporal_tolerance is not given. Estimate on half the median time step of the model data.
        if temporal_tolerance is None:
            median_dt = self.time.unique().to_series().diff().median()
            temporal_tolerance = median_dt / 2

        matched_times = self._match_to_nearest_times(
            vo.data[["z"]].to_dataframe(),
            t_tol=temporal_tolerance,
        )

        pairs = self._interpolate_to_obs_depths(
            vo.data.to_dataframe(),
            matched_times["obs_time"],
            matched_times["mod_time"],
            mod_value_col=self.name,
        )
        # Convert to xarray Dataset and set kind attribute
        xarr = pairs.reset_index().set_index(["time"]).to_xarray()
        xarr[self.name].attrs["kind"] = "model"

        return xarr
