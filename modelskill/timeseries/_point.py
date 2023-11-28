from __future__ import annotations
from pathlib import Path
from typing import get_args, Optional
import pandas as pd
import xarray as xr

import mikeio

from ..types import GeometryType, PointType, Quantity
from ..utils import _get_name
from ._timeseries import _validate_data_var_name


def _parse_point_input(
    data: PointType,
    name: Optional[str],
    item: str | int | None,
    quantity: Optional[Quantity],
) -> xr.Dataset:
    """Convert accepted input data to an xr.Dataset"""
    assert isinstance(
        data, get_args(PointType)
    ), f"Could not construct object from provided data of type {type(data)}"

    if isinstance(data, (str, Path)):
        assert Path(data).suffix == ".dfs0", "File must be a dfs0 file"
        name = name or Path(data).stem
        data = mikeio.read(data)  # now mikeio.Dataset
    elif isinstance(data, mikeio.Dfs0):
        data = data.read()  # now mikeio.Dataset

    # parse items
    if isinstance(data, (mikeio.DataArray, pd.Series, xr.DataArray)):
        item_name = data.name if hasattr(data, "name") else "PointModelResult"
        if item is not None:
            raise ValueError(f"item must be None when data is a {type(data)}")
    elif isinstance(data, (mikeio.Dataset, pd.DataFrame, xr.Dataset)):
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
        time_dim_name = list(data.dims)[0]
        if time_dim_name != "time":
            data = data.rename({time_dim_name: "time"})
        ds = data

    name = name or item_name
    name = _validate_data_var_name(name)

    # basic processing
    ds = ds.dropna(dim="time")

    vars = [v for v in ds.data_vars]
    assert len(ds.data_vars) == 1
    ds = ds.rename({vars[0]: name})

    ds[name].attrs["long_name"] = model_quantity.name
    ds[name].attrs["units"] = model_quantity.unit

    ds.attrs["gtype"] = str(GeometryType.POINT)
    assert isinstance(ds, xr.Dataset)
    return ds
