from __future__ import annotations
from typing import Optional

import xarray as xr

from ..types import TrackType, Quantity
from ..timeseries import TimeSeries, _parse_track_input


class TrackModelResult(TimeSeries):
    """Construct a TrackModelResult from a dfs0 file,
    mikeio.Dataset or pandas.DataFrame

    Parameters
    ----------
    data : types.TrackType
        the input data or file path
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
    quantity : Optional[str], optional
        A string to identify the quantity, by default None
    keep_duplicates : (str, bool), optional
        strategy for handling duplicate timestamps (wraps xarray.Dataset.drop_duplicates)
        "first" to keep first occurrence, "last" to keep last occurrence,
        False to drop all duplicates, "offset" to add milliseconds to
        consecutive duplicates, by default "first"
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
            )

        assert isinstance(data, xr.Dataset)
        data_var = str(list(data.data_vars)[0])
        data[data_var].attrs["kind"] = "model"
        super().__init__(data=data)

    def extract(self, obs) -> TrackModelResult:
        # TODO check x,y,z
        return self
