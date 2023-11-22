from __future__ import annotations
from typing import Optional

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
    ) -> None:
        ds = _parse_track_input(
            data=data,
            name=name,
            item=item,
            quantity=quantity,
            x_item=x_item,
            y_item=y_item,
        )
        data_var = str(list(ds.data_vars)[0])
        ds[data_var].attrs["kind"] = "model"
        super().__init__(data=ds)
