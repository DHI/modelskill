from __future__ import annotations
from typing import Optional, Sequence

import xarray as xr

from ..obs import TrackObservation
from ..types import TrackType
from ..quantity import Quantity
from ..timeseries import TimeSeries, _parse_track_input


class TrackModelResult(TimeSeries):
    """Construct a TrackModelResult from a dfs0 file,
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
        keep_duplicates: str | bool = "first",
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

    def extract(
        self, obs: TrackObservation, spatial_method: Optional[str] = None
    ) -> TrackModelResult:
        if not isinstance(obs, TrackObservation):
            raise ValueError(f"obs must be a TrackObservation not {type(obs)}")
        if spatial_method is not None:
            raise NotImplementedError(
                "spatial interpolation not possible when matching track model results with track observations"
            )
        # TODO check x,y,z
        return self
