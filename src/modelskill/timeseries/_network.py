from __future__ import annotations
from typing import Optional, Sequence
import xarray as xr

from ..types import GeometryType, NetworkType
from ..quantity import Quantity


def _parse_network_input(
    data: NetworkType,  # Which types should this accept? Res1D would make mikeio1d a dependency...
    name: Optional[str],
    item: str | int | None,
    quantity: Optional[Quantity],
    aux_items: Optional[Sequence[int | str]] = None,
) -> xr.Dataset:
    assert isinstance(
        data, NetworkType
    ), "Could not construct NetworkModelResult from provided data."

    # parse items
    valid_items = list(data.columns)

    # parse quantity
    model_quantity = Quantity.undefined() if quantity is None else quantity

    # select relevant items and convert to xr.Dataset
    relevant_items = valid_items + aux_items
    data = data[relevant_items]

    ds = data.to_xarray()
    ds = ds.dropna(dim="time")  # Drop NaNs from all data variables
    ds[name].attrs["long_name"] = model_quantity.name
    ds[name].attrs["units"] = model_quantity.unit

    ds = ds.set_coords("node")
    for aux_item in aux_items:
        ds[aux_item].attrs["kind"] = "aux"

    ds.attrs["gtype"] = str(GeometryType.NETWORK)
    assert isinstance(ds, xr.Dataset)
    return ds
