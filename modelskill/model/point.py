from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..types import Quantity, PointType
from ..timeseries import TimeSeries, _parse_point_input

if TYPE_CHECKING:
    import xarray as xr


class PointModelResult(TimeSeries):
    """Construct a PointModelResult from a dfs0 file,
    mikeio.Dataset/DataArray or pandas.DataFrame/Series

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
    ) -> None:
        if not self._is_input_validated(data):
            data = _parse_point_input(data, name=name, item=item, quantity=quantity)

            data.coords["x"] = x
            data.coords["y"] = y
            data.coords["z"] = None  # TODO: or np.nan?

        assert isinstance(data, xr.Dataset)

        data_var = str(list(data.data_vars)[0])
        data[data_var].attrs["kind"] = "model"
        super().__init__(data=data)
