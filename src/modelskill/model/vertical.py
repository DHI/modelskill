from __future__ import annotations
from typing import Any, Literal, Optional, Sequence

import xarray as xr

from ..types import VerticalType
from ..quantity import Quantity
from ..timeseries import TimeSeries, _parse_vertical_input


class VerticalModelResult(TimeSeries):
    """Model result for a vertical column.

    Construct a VerticalColumnModelResult from a dfs0 file,
    mikeio.Dataset, pandas.DataFrame or a xarray.Datasets

    Parameters
    ----------
    data : str, Path, pd.DataFrame, mikeio.Dfs0, mikeio.Dfs0, xr.Dataset
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

    @property
    def z(self) -> Any:
        """z-coordinate"""
        return self._coordinate_values("z")
