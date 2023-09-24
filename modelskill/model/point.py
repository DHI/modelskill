from __future__ import annotations
from pathlib import Path
from typing import Optional, get_args
import mikeio
import pandas as pd
import xarray as xr

from ..utils import _get_name
from ..types import Quantity, PointType
from ..timeseries import TimeSeries  # TODO move to main module


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
        name: Optional[str] = None,  # TODO should maybe be required?
        x: Optional[float] = None,
        y: Optional[float] = None,
        item: str | int | None = None,
        quantity: Optional[Quantity] = None,
    ) -> None:
        assert isinstance(
            data, get_args(PointType)
        ), "Could not construct PointModelResult from provided data"

        if isinstance(data, (str, Path)):
            assert Path(data).suffix == ".dfs0", "File must be a dfs0 file"
            name = name or Path(data).stem
            data = mikeio.read(data)  # now mikeio.Dataset
        elif isinstance(data, mikeio.Dfs0):
            data = data.read()  # now mikeio.Dataset

        # parse items
        if isinstance(data, (mikeio.DataArray, pd.Series, xr.DataArray)):  # type: ignore
            item_name = data.name if hasattr(data, "name") else "PointModelResult"
            if item is not None:
                raise ValueError(f"item must be None when data is a {type(data)}")
        elif isinstance(data, (mikeio.Dataset, pd.DataFrame, xr.Dataset)):  # type: ignore
            if isinstance(data, mikeio.Dataset):
                valid_items = [i.name for i in data.items]
            elif isinstance(data, pd.DataFrame):
                valid_items = list(data.columns)
            else:
                valid_items = list(data.data_vars)
            item_name = _get_name(x=item, valid_names=valid_items)

        # select relevant items
        if isinstance(data, mikeio.DataArray):
            data = mikeio.Dataset(data)
        elif isinstance(data, pd.Series):
            data = data.to_frame()
        elif isinstance(data, xr.DataArray):
            data = data.to_dataset()
        else:
            data = data[[item_name]]
        assert isinstance(data, (mikeio.Dataset, pd.DataFrame, xr.Dataset))

        # parse quantity
        if isinstance(data, mikeio.Dataset):
            if quantity is None:
                quantity = Quantity.from_mikeio_iteminfo(data[0].item)
        model_quantity = Quantity.undefined() if quantity is None else quantity

        # convert to xr.Dataset
        if isinstance(data, mikeio.Dataset):
            ds = data.to_xarray()
        elif isinstance(data, pd.DataFrame):
            data.index.name = "time"
            ds = data.to_xarray()
        else:
            assert len(data.dims) == 1, "Only 0-dimensional data are supported"
            if data.coords[list(data.coords)[0]].name != "time":
                data = data.rename({list(data.coords)[0]: "time"})
            ds = data

        name = name or item_name
        assert isinstance(name, str)

        # basic processing
        ds = ds.dropna(dim="time")
        if len(ds) == 0 or len(ds["time"]) == 0:
            raise ValueError("No data.")

        # df.index = make_unique_index(df.index, offset_duplicates=0.001)
        if not ds.time.to_index().is_monotonic_increasing:
            # TODO: duplicates_keep="mean","first","last"
            raise ValueError(
                "Time axis has duplicate entries. It must be monotonically increasing."
            )

        super().__init__(data=ds, name=name, quantity=model_quantity)
        self.x = x
        self.y = y

        # self.gtype = GeometryType.POINT
