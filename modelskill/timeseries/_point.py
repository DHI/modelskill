from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, get_args, List, Optional
import numpy as np
import pandas as pd
import xarray as xr

import mikeio

from ..types import GeometryType, PointType
from ..quantity import Quantity
from ..utils import _get_name
from ._timeseries import _validate_data_var_name


@dataclass
class PointItem:
    values: str
    aux: list[str]

    @property
    def all(self) -> List[str]:
        return [self.values] + self.aux


def _parse_point_items(
    items: Sequence[str],
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


def _parse_point_input(
    data: PointType,
    name: Optional[str],
    item: str | int | None,
    quantity: Optional[Quantity],
    x: Optional[float],
    y: Optional[float],
    z: Optional[float],
    aux_items: Optional[Sequence[int | str]],
) -> xr.Dataset:
    """Convert accepted input data to an xr.Dataset"""
    assert isinstance(
        data, get_args(PointType)
    ), f"Could not construct object from provided data of type {type(data)}"

    if isinstance(data, (str, Path)):
        suffix = Path(data).suffix
        name = name or Path(data).stem
        if suffix == ".dfs0":
            data = mikeio.read(data)  # now mikeio.Dataset
        elif suffix == ".nc":
            data = xr.open_dataset(data)
    elif isinstance(data, mikeio.Dfs0):
        data = data.read()  # now mikeio.Dataset

    # parse items
    if isinstance(data, (mikeio.DataArray, pd.Series, xr.DataArray)):
        item_name = data.name if hasattr(data, "name") else "PointModelResult"
        if item is not None:
            raise ValueError(f"item must be None when data is a {type(data)}")
        if aux_items is not None:
            raise ValueError(f"aux_items must be None when data is a {type(data)}")
        sel_items = PointItem(values=item_name, aux=[])

        if isinstance(data, mikeio.DataArray):
            data = mikeio.Dataset(data)
        elif isinstance(data, pd.Series):
            data = data.to_frame()
        elif isinstance(data, xr.DataArray):
            data = data.to_dataset()

    elif isinstance(data, (mikeio.Dataset, pd.DataFrame, xr.Dataset)):
        if isinstance(data, mikeio.Dataset):
            valid_items = [i.name for i in data.items]
        elif isinstance(data, pd.DataFrame):
            valid_items = list(data.columns)
        else:
            valid_items = list(data.data_vars)
        sel_items = _parse_point_items(valid_items, item=item, aux_items=aux_items)
        item_name = sel_items.values
        data = data[sel_items.all]
    else:
        raise ValueError("Could not Point object from provided data")

    assert isinstance(data, (mikeio.Dataset, pd.DataFrame, xr.Dataset))

    # parse quantity
    if isinstance(data, mikeio.Dataset):
        if quantity is None:
            quantity = Quantity.from_mikeio_iteminfo(data[0].item)

    if isinstance(data, xr.Dataset):
        if quantity is None:
            da = data[sel_items.values]
            quantity = Quantity.from_cf_attrs(da.attrs)
    model_quantity = Quantity.undefined() if quantity is None else quantity

    # convert to xr.Dataset
    if isinstance(data, mikeio.Dataset):
        ds = data.to_xarray()
    elif isinstance(data, pd.DataFrame):
        data.index.name = "time"
        ds = data.to_xarray()
    else:
        assert len(data.dims) == 1, "Only 0-dimensional data are supported"
        time_dim_name = list(data.dims)[0]
        if time_dim_name != "time":
            data = data.rename({time_dim_name: "time"})
        ds = data

    assert isinstance(ds, xr.Dataset)

    name = name or item_name
    name = _validate_data_var_name(name)

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

    ds[name].attrs["long_name"] = model_quantity.name
    ds[name].attrs["units"] = model_quantity.unit
    ds[name].attrs["is_directional"] = int(model_quantity.is_directional)

    for aux_item in sel_items.aux:
        ds[aux_item].attrs["kind"] = "aux"

    ds.attrs["gtype"] = str(GeometryType.POINT)

    coords2d = {"x": x, "y": y}
    for coord, value in coords2d.items():
        if value is not None:
            ds.coords[coord] = value

        if coord not in ds.coords:
            ds.coords[coord] = np.nan

    ds.coords["z"] = z

    assert isinstance(ds, xr.Dataset)
    return ds
