from __future__ import annotations
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, get_args, List, Optional, Tuple
import pandas as pd
import xarray as xr

import warnings
import mikeio
import mikeio1d

from ..types import GeometryType, PointType
from ..quantity import Quantity
from ..utils import _get_name
from ._timeseries import _validate_data_var_name
from ._coords import XYZCoords, NetworkCoords


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


def _parse_items(
    data: PointType,
    item: Optional[str | None] = None,
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
        raise ValueError("Could not Point object from provided data")

    return sel_items


def _convert_to_dataset(
    data: PointType, varname: str, sel_items: PointItem
) -> xr.Dataset:
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

    assert isinstance(ds, xr.Dataset)

    return ds


def _include_coords(
    ds: xr.Dataset,
    *,
    coords: Optional[XYZCoords] = None,
    network_coords: Optional[NetworkCoords] = None,
) -> xr.Dataset:
    ds = ds.copy()
    if coords is not None:
        # ds might already have some coordinates set, so we will update
        # only the ones that are NOT already present
        new_coords = set(coords.as_dict).difference(ds.coords)
        incoming_coords = {k: coords.as_dict[k] for k in new_coords}
        ds.coords.update(incoming_coords)
    if network_coords is not None:
        ds.coords.update(network_coords.as_dict)

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
        ds[aux_item].attrs["kind"] = "aux"

    return ds


def _open_and_name(data: PointType, name: Optional[str]) -> Tuple[PointType, str]:
    assert isinstance(
        data, get_args(PointType)
    ), f"Could not construct object from provided data of type {type(data)}"

    if isinstance(data, (str, Path)):
        suffix = Path(data).suffix
        if suffix == ".dfs0":
            name = name or Path(data).stem
            data = mikeio.read(data)  # now mikeio.Dataset
        elif suffix == ".nc":
            stem = Path(data).stem
            data = xr.open_dataset(data)
            name = name or data.attrs.get("name") or stem
        elif suffix == ".res1d":
            name = name or Path(data).stem
            data = mikeio1d.open(data)
    elif isinstance(data, mikeio.Dfs0):
        data = data.read()  # now mikeio.Dataset
    elif isinstance(data, mikeio1d.Res1D):
        data = data.read()  # now mikeio1d.Res1D

    return data, name


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

    data, name = _open_and_name(data, name)
    sel_items = _parse_items(data, item, aux_items)

    if isinstance(data, mikeio.DataArray):
        data = mikeio.Dataset([data])
    elif isinstance(data, pd.Series):
        data = data.to_frame()
    elif isinstance(data, xr.DataArray):
        data = data.to_dataset()
    elif isinstance(data, (mikeio.Dataset, pd.DataFrame, xr.Dataset)):
        data = data[sel_items.all]
    else:
        raise ValueError(f"Invalid data type {type(data)}")

    varname = name or sel_items.values
    if not isinstance(varname, str):
        raise ValueError(f"Invalid variable name: {varname}")
    ds = _convert_to_dataset(data, varname, sel_items)

    # parse quantity
    if isinstance(data, mikeio.Dataset):
        if quantity is None:
            quantity = Quantity.from_mikeio_iteminfo(data[0].item)

    if isinstance(data, xr.Dataset):
        if quantity is None:
            da = data[sel_items.values]
            quantity = Quantity.from_cf_attrs(da.attrs)
    model_quantity = Quantity.undefined() if quantity is None else quantity

    ds = _include_attributes(ds, varname, model_quantity, sel_items)

    coords = XYZCoords(x, y, z)
    ds = _include_coords(ds, coords=coords)

    return ds


def _parse_network_input(
    data: mikeio1d.Res1D | str,
    variable: Optional[str] = None,
    *,
    node: Optional[int] = None,
    reach: Optional[str] = None,
    chainage: Optional[str | float] = None,
    gridpoint: Optional[int | Literal["start", "end"]] = None,
) -> pd.Series:
    def variable_name_to_res1d(name: str) -> str:
        return name.replace(" ", "").replace("_", "")

    if isinstance(data, (str, Path)):
        data = mikeio1d.open(data)

    if ("reaches" not in dir(data)) or ("nodes" not in dir(data)):
        raise ValueError(
            "Invalid file format. Data must have a network structure containing 'nodes' and 'reaches'."
        )

    by_node = node is not None
    by_reach = reach is not None
    with_chainage = chainage is not None
    with_index = gridpoint is not None

    if by_node and not by_reach:
        location = data.nodes[str(node)]
        if with_chainage or with_index:
            warnings.warn(
                "'chainage' or 'gridpoint' are only relevant when passed with 'reach' but they were passed with 'node', so they will be ignored."
            )

    elif by_reach and not by_node:
        location = data.reaches[reach]
        if with_index == with_chainage:
            raise ValueError(
                "Locations accessed by chainage must be specified either by chainage or by index, not both."
            )

        if with_index and not with_chainage:
            gridpoint = 0 if gridpoint == "start" else gridpoint
            gridpoint = -1 if gridpoint == "end" else gridpoint
            chainage = location.chainages[gridpoint]

        location = location[chainage]

    else:
        raise ValueError(
            "A network location must be specified either by node or by reach."
        )

    if variable is None:
        if len(location.quantities) != 1:
            raise ValueError(
                f"The network location does not have a unique quantity: {location.quantities}, in such case 'variable' argument cannot be None"
            )
        res1d_name = location.quantities[0]
    else:
        # After filtering by node or by reach and chainage, a location will only
        # have unique quantities
        res1d_name = variable_name_to_res1d(variable)
    df = location.to_dataframe()
    if df.shape[1] == 1:
        colname = df.columns[0]
        if res1d_name not in colname:
            raise ValueError(f"Column name '{colname}' does not contain '{res1d_name}'")

        return df.rename(columns={colname: res1d_name})[res1d_name].copy()
    else:
        raise ValueError(f"Multiple matching quantites found at location: {df.columns}")
