from __future__ import annotations
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args, Optional, List, Sequence
import warnings
import pandas as pd
import xarray as xr
from ._coords import XYZCoords
import numpy as np

import mikeio

from ..types import GeometryType, VerticalType
from ..quantity import Quantity
from ..utils import _get_name
from ._timeseries import _validate_data_var_name


@dataclass
class VerticalItem:
    z: str
    values: str
    aux: list[str]

    @property
    def all(self) -> List[str]:
        return [self.z, self.values] + self.aux


def _parse_vertical_items(
    items: Sequence[Hashable],
    z_item: int | str | None,
    item: int | str | None,
    aux_items: Optional[Sequence[int | str]] = None,
) -> VerticalItem:
    """If input has exactly 2 items we accept item=None"""
    if len(items) < 2:
        raise ValueError(
            f"Input has only {len(items)} items. It should have at least 2."
        )
    if item is None:
        if len(items) == 2:
            item = 1
        elif len(items) > 2:
            raise ValueError(
                f"Input has more than 2 items, but item was not given! Available items: {items}"
            )

    item = _get_name(item, valid_names=items)
    z_item = _get_name(z_item, valid_names=items)
    if isinstance(aux_items, (str, int)):
        aux_items = [aux_items]
    aux_items_str = [_get_name(i, valid_names=items) for i in aux_items or []]

    # check that there are no duplicates
    res = VerticalItem(z=z_item, values=item, aux=aux_items_str)
    if len(set(res.all)) != len(res.all):
        raise ValueError(f"Duplicate items! {res.all}")

    return res


def _include_location(
    ds: xr.Dataset,
    *,
    coords: Optional[XYZCoords] = None,
) -> xr.Dataset:
    ds = ds.copy()
    if coords is not None:
        # ds might already have some coordinates set, so we will update
        # only the ones that are NOT already present
        coords_to_add = {}
        for k, v in coords.as_dict.items():
            # Add if coordinate doesn't exist, or if user provided a non-null value
            if k not in ds.coords or (v is not None and not np.isnan(v)):
                coords_to_add[k] = v
        ds.coords.update(coords_to_add)

    return ds


def _parse_vertical_input(
    data: VerticalType,
    name: Optional[str],
    item: str | int | None,
    quantity: Optional[Quantity],
    z_item: str | int | None,
    x: float | None = None,
    y: float | None = None,
    keep_duplicates: Literal["first", "last", False] = "first",
    aux_items: Optional[Sequence[int | str]] = None,
) -> xr.Dataset:
    assert isinstance(
        data, get_args(VerticalType)
    ), "Could not construct Vertical object from provided data."
    if isinstance(data, (str, Path)):
        if Path(data).suffix != ".dfs0":
            raise ValueError(f"File must be a dfs0 file, not {Path(data).suffix}")
        name = name or Path(data).stem
        data = mikeio.read(data)  # now mikeio.Dataset
    elif isinstance(data, mikeio.Dfs0):
        data = data.read()  # now mikeio.Dataset

    # parse items
    valid_items: Sequence[Hashable]
    if isinstance(data, mikeio.Dataset):
        valid_items = [i.name for i in data.items]
    elif isinstance(data, pd.DataFrame):
        valid_items = list(data.columns)
    elif isinstance(data, xr.Dataset):
        valid_items = list(data.data_vars)
    else:
        raise ValueError("Could not construct Vertical object from provided data")

    sel_items = _parse_vertical_items(
        valid_items, z_item=z_item, item=item, aux_items=aux_items
    )

    name = name or sel_items.values
    name = _validate_data_var_name(name)

    # parse quantity
    if isinstance(data, mikeio.Dataset):
        if quantity is None:
            quantity = Quantity.from_mikeio_iteminfo(data[sel_items.values].item)
    model_quantity = Quantity.undefined() if quantity is None else quantity

    # Convert to xr.Dataset and select relevant items and
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

    ds = ds.rename({sel_items.z: "z"})

    # keep first, last or none of duplicate (time, z) pairs
    idx_df = pd.DataFrame({"time": ds["time"].to_index(), "z": ds["z"].values})

    keep_mask = ~idx_df.duplicated(subset=["time", "z"], keep=keep_duplicates)

    n_removed = int((~keep_mask).sum())
    ds = ds.isel(time=keep_mask.values)
    if n_removed > 0:
        warnings.warn(
            f"Removed {n_removed} duplicate (time, z) entries with keep={keep_duplicates}"
        )

    ds = ds.dropna(dim="time", subset=["z"])  # remove times with z as nan

    SPATIAL_DIMS = ["x", "y", "z"]
    for dim in SPATIAL_DIMS:
        if dim in ds:
            ds = ds.set_coords(dim)

    # Fixed x,y location
    coords = XYZCoords(x=x, y=y, z=None)
    ds = _include_location(ds, coords=coords)  # Add x and y

    assert len(ds.data_vars) == 1 + len(sel_items.aux)
    data_var = str(list(ds.data_vars)[0])
    ds = ds.rename({data_var: name})
    ds[name].attrs["long_name"] = model_quantity.name
    ds[name].attrs["units"] = model_quantity.unit

    for aux_item in sel_items.aux:
        ds[aux_item].attrs["kind"] = "aux"

    ds.attrs["gtype"] = str(GeometryType.VERTICAL)
    assert isinstance(ds, xr.Dataset)
    return ds
