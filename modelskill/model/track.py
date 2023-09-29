from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import get_args, Optional, List
import pandas as pd
import xarray as xr

import mikeio

from ..types import GeometryType, TrackType, Quantity
from ..timeseries import TimeSeries

from ..utils import make_unique_index, get_item_name_and_idx


@dataclass
class TrackItem:
    x: str
    y: str
    values: str

    @property
    def all(self) -> List[str]:
        return [self.x, self.y, self.values]


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
        assert isinstance(
            data, get_args(TrackType)
        ), "Could not construct TrackModelResult from provided data."

        if isinstance(data, (str, Path)):
            assert Path(data).suffix == ".dfs0", "File must be a dfs0 file"
            name = name or Path(data).stem
            data = mikeio.read(data)  # now mikeio.Dataset
        elif isinstance(data, mikeio.Dfs0):
            data = data.read()  # now mikeio.Dataset

        # parse items
        if isinstance(data, mikeio.Dataset):
            valid_items = [i.name for i in data.items]
        elif isinstance(data, pd.DataFrame):
            valid_items = list(data.columns)
        elif isinstance(data, xr.Dataset):
            valid_items = list(data.data_vars)
        else:
            raise ValueError("Could not construct TrackModelResult from provided data")

        ti = self._parse_track_items(valid_items, x_item, y_item, item)
        name = name or ti.values
        name = self._validate_name(name)

        # parse quantity
        if isinstance(data, mikeio.Dataset):
            if quantity is None:
                quantity = Quantity.from_mikeio_iteminfo(data[ti.values].item)
        model_quantity = Quantity.undefined() if quantity is None else quantity

        # select relevant items and convert to xr.Dataset
        assert isinstance(data, (mikeio.Dataset, pd.DataFrame, xr.Dataset))
        data = data[ti.all]
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

        ds = ds.rename({ti.x: "x", ti.y: "y"})
        ds = ds.dropna(dim="time", subset=["x", "y"])
        ds["time"] = make_unique_index(ds["time"].to_index(), offset_duplicates=0.001)

        # ds["z"] = None  # TODO: or np.nan?

        SPATIAL_DIMS = ["x", "y", "z"]

        for dim in SPATIAL_DIMS:
            if dim in ds:
                ds = ds.set_coords(dim)

        assert len(ds.data_vars) == 1
        data_var = str(list(ds.data_vars)[0])
        ds = ds.rename({data_var: name})
        ds[name].attrs["kind"] = "model"
        # ds[name].attrs["quantity"] = model_quantity.to_dict()
        ds[name].attrs["long_name"] = model_quantity.name
        ds[name].attrs["units"] = model_quantity.unit

        ds.attrs["gtype"] = str(GeometryType.TRACK)

        super().__init__(data=ds)

    @staticmethod
    def _parse_track_items(items, x_item, y_item, item) -> TrackItem:
        """If input has exactly 3 items we accept item=None"""
        if len(items) < 3:
            raise ValueError(
                f"Input has only {len(items)} items. It should have at least 3."
            )
        if item is None:
            if len(items) == 3:
                item = 2
            elif len(items) > 3:
                raise ValueError("Input has more than 3 items, but item was not given!")

        item, _ = get_item_name_and_idx(items, item)
        x_item, _ = get_item_name_and_idx(items, x_item)
        y_item, _ = get_item_name_and_idx(items, y_item)

        if (item == x_item) or (item == y_item) or (x_item == y_item):
            raise ValueError(
                f"x-item ({x_item}), y-item ({y_item}) and value-item ({item}) must be different!"
            )
        # return x_item, y_item, item
        return TrackItem(x=x_item, y=y_item, values=item)
