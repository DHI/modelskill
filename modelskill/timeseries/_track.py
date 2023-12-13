from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import get_args, Optional, List, Sequence
import warnings
import pandas as pd
import xarray as xr

import mikeio

from ..types import GeometryType, TrackType
from ..quantity import Quantity
from ..utils import _get_name, make_unique_index
from ._timeseries import _validate_data_var_name


@dataclass
class TrackItem:
    x: str
    y: str
    values: str
    aux: list[str]

    @property
    def all(self) -> List[str]:
        return [self.x, self.y, self.values] + self.aux


def _parse_track_items(
    items: Sequence[str],
    x_item: int | str | None,
    y_item: int | str | None,
    item: int | str | None,
    aux_items: Optional[Sequence[int | str]] = None,
) -> TrackItem:
    """If input has exactly 3 items we accept item=None"""
    if len(items) < 3:
        raise ValueError(
            f"Input has only {len(items)} items. It should have at least 3."
        )
    if item is None:
        if len(items) == 3:
            item = 2
        elif len(items) > 3:
            raise ValueError(
                f"Input has more than 3 items, but item was not given! Available items: {items}"
            )

    item = _get_name(item, valid_names=items)
    x_item = _get_name(x_item, valid_names=items)
    y_item = _get_name(y_item, valid_names=items)
    if isinstance(aux_items, (str, int)):
        aux_items = [aux_items]
    aux_items_str = [_get_name(i, valid_names=items) for i in aux_items or []]

    # check that there are no duplicates
    res = TrackItem(x=x_item, y=y_item, values=item, aux=aux_items_str)
    if len(set(res.all)) != len(res.all):
        raise ValueError(f"Duplicate items! {res.all}")

    return res


def _parse_track_input(
    data: TrackType,
    name: Optional[str],
    item: str | int | None,
    quantity: Optional[Quantity],
    x_item: str | int | None,
    y_item: str | int | None,
    keep_duplicates: bool | str,
    offset_duplicates: float = 0.001,
    aux_items: Optional[Sequence[int | str]] = None,
) -> xr.Dataset:
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
        raise ValueError("Could not construct Track object from provided data")

    sel_items = _parse_track_items(
        valid_items, x_item=x_item, y_item=y_item, item=item, aux_items=aux_items
    )
    name = name or sel_items.values
    name = _validate_data_var_name(name)

    # parse quantity
    if isinstance(data, mikeio.Dataset):
        if quantity is None:
            quantity = Quantity.from_mikeio_iteminfo(data[sel_items.values].item)
    model_quantity = Quantity.undefined() if quantity is None else quantity

    # select relevant items and convert to xr.Dataset
    assert isinstance(data, (mikeio.Dataset, pd.DataFrame, xr.Dataset))
    data = data[sel_items.all]
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

    ds = ds.rename({sel_items.x: "x", sel_items.y: "y"})

    # A unique index makes lookup much faster O(1)
    if keep_duplicates == "offset":
        ds["time"] = make_unique_index(
            ds["time"].to_index(), offset_duplicates=offset_duplicates
        )
    else:
        # keep first, last or none of the duplicates
        n = len(ds["time"])
        ds = ds.drop_duplicates(dim="time", keep=keep_duplicates)
        n_removed = n - len(ds["time"])
        if n_removed > 0:
            warnings.warn(
                f"Removed {n_removed} duplicate timestamps with keep={keep_duplicates}"
            )
    ds = ds.dropna(dim="time", subset=["x", "y"])

    SPATIAL_DIMS = ["x", "y", "z"]
    for dim in SPATIAL_DIMS:
        if dim in ds:
            ds = ds.set_coords(dim)

    assert len(ds.data_vars) == 1 + len(sel_items.aux)
    data_var = str(list(ds.data_vars)[0])
    ds = ds.rename({data_var: name})
    ds[name].attrs["long_name"] = model_quantity.name
    ds[name].attrs["units"] = model_quantity.unit

    for aux_item in sel_items.aux:
        ds[aux_item].attrs["kind"] = "aux"

    ds.attrs["gtype"] = str(GeometryType.TRACK)
    assert isinstance(ds, xr.Dataset)
    return ds
