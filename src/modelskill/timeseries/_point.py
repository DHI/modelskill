from __future__ import annotations
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, get_args, List, Optional, Tuple, Union, Any
import pandas as pd
import xarray as xr
import numpy as np

import mikeio

from ..types import GeometryType, PointType, VariableKind
from ..quantity import Quantity
from ..utils import _get_name
from ._timeseries import _validate_data_var_name
from ._coords import XYZCoords


@dataclass
class PointItem:
    values: str
    aux: list[str]

    @property
    def all(self) -> List[str]:
        return [self.values] + self.aux


def _parse_point_items(
    items: Sequence[Hashable],
    item: int | str | None,
    aux_items: Optional[Sequence[int | str]] = None,
) -> PointItem:
    """If input has exactly 1 item we accept item=None"""
    if item is None:
        if len(items) == 1:
            item = 0
        elif len(items) > 1:
            raise ValueError(
                f"Input has more than 1 item, but item was not given! Available items: {items}"
            )

    item = _get_name(item, valid_names=items)
    if isinstance(aux_items, (str, int)):
        aux_items = [aux_items]
    aux_items_str = [_get_name(i, valid_names=items) for i in aux_items or []]

    # check that there are no duplicates
    res = PointItem(values=item, aux=aux_items_str)
    if len(set(res.all)) != len(res.all):
        raise ValueError(f"Duplicate items! {res.all}")

    return res


def _select_items(
    data: Union[
        mikeio.Dataset,
        mikeio.DataArray,
        xr.Dataset,
        xr.DataArray,
        pd.Series,
        pd.DataFrame,
    ],
    item: Optional[str | int] = None,
    aux_items: Optional[Sequence[int | str]] = None,
) -> PointItem:
    if isinstance(data, (mikeio.DataArray, pd.Series, xr.DataArray)):
        item_name = data.name if data.name is not None else "PointModelResult"
        assert isinstance(item_name, str)
        if item is not None:
            raise ValueError(f"item must be None when data is a {type(data)}")
        if aux_items is not None:
            raise ValueError(f"aux_items must be None when data is a {type(data)}")
        sel_items = PointItem(values=str(item_name), aux=[])

    elif isinstance(data, (mikeio.Dataset, pd.DataFrame, xr.Dataset)):
        valid_items: Sequence[Hashable]
        if isinstance(data, mikeio.Dataset):
            valid_items = [i.name for i in data.items]
        elif isinstance(data, pd.DataFrame):
            valid_items = list(data.columns)
        else:
            valid_items = list(data.data_vars)
        sel_items = _parse_point_items(valid_items, item=item, aux_items=aux_items)
    else:
        raise ValueError("Could not create Point object from provided data")

    return sel_items


def _convert_to_dataset(
    data: Union[
        pd.DataFrame,
        pd.Series,
        mikeio.Dataset,
        mikeio.DataArray,
        xr.Dataset,
        xr.DataArray,
    ],
    varname: str,
    sel_items: PointItem,
) -> xr.Dataset:
    if isinstance(data, mikeio.DataArray):
        data = mikeio.Dataset([data])
    elif isinstance(data, pd.Series):
        data = data.to_frame()
    elif isinstance(data, xr.DataArray):
        data = data.to_dataset()
    elif isinstance(data, (mikeio.Dataset, pd.DataFrame, xr.Dataset)):
        data = data[sel_items.all]

    if isinstance(data, mikeio.Dataset):
        ds = data.to_xarray()
    elif isinstance(data, pd.DataFrame):
        data.index.name = "time"
        ds = data.to_xarray()
    elif isinstance(data, xr.Dataset):
        assert len(data.dims) == 1, "Only 0-dimensional data are supported"
        time_dim_name = list(data.dims)[0]
        if time_dim_name != "time":
            data = data.rename({time_dim_name: "time"})
        ds = data

    name = _validate_data_var_name(varname)

    n_unique_times = len(ds.time.to_index().unique())
    if n_unique_times < len(ds.time):
        raise ValueError("time must be unique (please check for duplicate times))")

    if not ds.time.to_index().is_monotonic_increasing:
        raise ValueError("time must be increasing (please check for duplicate times))")

    # basic processing
    ds = ds.dropna(dim="time")

    vars = [v for v in ds.data_vars]
    assert len(ds.data_vars) == 1 + len(sel_items.aux)
    ds = ds.rename({vars[0]: name})

    return ds


def _include_coords(
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


def _include_attributes(
    ds: xr.Dataset, name: str, quantity: Quantity, sel_items: PointItem
) -> xr.Dataset:
    ds = ds.copy()

    ds.attrs["gtype"] = str(GeometryType.POINT)

    ds[name].attrs["long_name"] = quantity.name
    ds[name].attrs["units"] = quantity.unit
    ds[name].attrs["is_directional"] = int(quantity.is_directional)

    for aux_item in sel_items.aux:
        ds[aux_item].attrs["kind"] = VariableKind.AUXILIARY.value

    return ds


def _open_and_name(
    data: PointType, name: Optional[str]
) -> Tuple[
    Union[
        mikeio.Dataset,
        mikeio.DataArray,
        xr.Dataset,
        xr.DataArray,
        pd.Series,
        pd.DataFrame,
    ],
    str,
]:
    assert isinstance(
        data, get_args(PointType)
    ), f"Could not construct object from provided data of type {type(data)}"

    name_is_arg = name is not None
    if isinstance(data, (str, Path)):
        suffix = Path(data).suffix
        stem = Path(data).stem
        name = name or stem
        if suffix == ".dfs0":
            data = mikeio.read(data)  # now mikeio.Dataset
        elif suffix == ".nc":
            data = xr.open_dataset(data)
            name = name if name_is_arg else data.attrs.get("name") or stem
    elif isinstance(data, mikeio.Dfs0):
        data = data.read()  # now mikeio.Dataset

    name = name or ""
    return data, name


def _get_quantity(data: Any, sel_items: PointItem) -> Quantity:
    if isinstance(data, mikeio.Dataset):
        return Quantity.from_mikeio_iteminfo(data[sel_items.values].item)
    if isinstance(data, mikeio.DataArray):
        return Quantity.from_mikeio_iteminfo(data.item)
    if isinstance(data, xr.Dataset):
        da = data[sel_items.values]
        return Quantity.from_cf_attrs(da.attrs)

    return Quantity.undefined()


def _select_variable_name(name: str, sel_items: PointItem) -> str:
    varname = name or sel_items.values
    if not isinstance(varname, str):
        raise ValueError(f"Invalid variable name: {varname}")
    return varname


def _parse_point_input(
    data: PointType,
    name: Optional[str],
    item: Optional[str | int],
    quantity: Optional[Quantity],
    aux_items: Optional[Sequence[int | str]],
    *,
    coords: XYZCoords,
) -> xr.Dataset:
    """Convert accepted input data to an xr.Dataset"""

    data, name = _open_and_name(data, name)
    sel_items = _select_items(data, item, aux_items)
    quantity = _get_quantity(data, sel_items) if quantity is None else quantity
    varname = _select_variable_name(name, sel_items)

    ds = _convert_to_dataset(data, varname, sel_items)
    ds = _include_attributes(ds, varname, quantity, sel_items)
    ds = _include_coords(ds, coords=coords)
    return ds


def _parse_xyz_point_input(
    data: PointType,
    name: Optional[str],
    item: str | int | None,
    quantity: Optional[Quantity],
    x: Optional[float],
    y: Optional[float],
    z: Optional[float],
    aux_items: Optional[Sequence[int | str]],
) -> xr.Dataset:
    coords = XYZCoords(x, y, z)
    ds = _parse_point_input(data, name, item, quantity, aux_items, coords=coords)
    return ds
