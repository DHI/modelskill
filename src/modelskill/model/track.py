from __future__ import annotations
from typing import Literal, Optional, Sequence
import warnings

import numpy as np
import xarray as xr

from ..types import TrackType
from ..obs import TrackObservation
from ..quantity import Quantity
from ..timeseries import TimeSeries, _parse_track_input


class TrackModelResult(TimeSeries):
    """Model result for a track.

    Construct a TrackModelResult from a dfs0 file,
    mikeio.Dataset, pandas.DataFrame or a xarray.Datasets

    Parameters
    ----------
    data : types.TrackType
        The input data or file path
    name : Optional[str], optional
        The name of the model result,
        by default None (will be set to file name or item name)
    item : str | int | None, optional
        If multiple items/arrays are present in the input an item
        must be given (as either an index or a string), by default None
    x_item : str | int | None, optional
        Item of the first coordinate of positions, by default None
    y_item : str | int | None, optional
        Item of the second coordinate of positions, by default None
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
        data: TrackType,
        *,
        name: Optional[str] = None,
        item: str | int | None = None,
        quantity: Optional[Quantity] = None,
        x_item: str | int = 0,
        y_item: str | int = 1,
        keep_duplicates: Literal["first", "last", False] = "first",
        aux_items: Optional[Sequence[int | str]] = None,
    ) -> None:
        if not self._is_input_validated(data):
            data = _parse_track_input(
                data=data,
                name=name,
                item=item,
                quantity=quantity,
                x_item=x_item,
                y_item=y_item,
                keep_duplicates=keep_duplicates,
                aux_items=aux_items,
            )

        assert isinstance(data, xr.Dataset)
        data_var = str(list(data.data_vars)[0])
        data[data_var].attrs["kind"] = "model"
        super().__init__(data=data)

    def subset_to(
        self, observation: TrackObservation, *, spatial_tolerance: float
    ) -> xr.Dataset:
        mri = self
        mod_df = mri.data.to_dataframe()
        obs_df = observation.data.to_dataframe()

        # 1. inner join on time
        df = mod_df.join(obs_df, how="inner", lsuffix="_mod", rsuffix="_obs")

        # 2. remove model points outside observation track
        n_points = len(df)
        keep_x = np.abs((df.x_mod - df.x_obs)) < spatial_tolerance
        keep_y = np.abs((df.y_mod - df.y_obs)) < spatial_tolerance
        df = df[keep_x & keep_y]
        if n_points_removed := n_points - len(df):
            warnings.warn(
                f"Removed {n_points_removed} model points outside observation track (spatial_tolerance={spatial_tolerance})"
            )
        return mri.data.sel(time=df.index)
