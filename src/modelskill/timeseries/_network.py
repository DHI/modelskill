from __future__ import annotations
from collections.abc import Hashable
from dataclasses import dataclass
from typing import get_args, Optional, List, Sequence
import xarray as xr

from ..types import GeometryType, NetworkType
from ..quantity import Quantity
from ..utils import _get_name
from ._timeseries import _validate_data_var_name


@dataclass
class NetworkItem:
    node: int
    values: str
    aux: list[str]

    def __post_init__(self):
        # Check for duplicates
        if len(set(self.all)) != len(self.all):
            raise ValueError(f"Duplicate items! {self.all}")

    @property
    def all(self) -> List[str]:
        return [self.node, self.values] + self.aux


def _parse_network_items(
    items: Sequence[Hashable],
    node_item: int
    | str
    | None,  # the item in the data that signals to the node -> is it necessary in our case?
    item: int | str | None,
    aux_items: Optional[Sequence[int | str]] = None,
) -> NetworkItem:
    item = _get_name(item, valid_names=items)
    node_item = _get_name(node_item, valid_names=items)
    if isinstance(aux_items, (str, int)):
        aux_items = [aux_items]
    aux_items_str = [_get_name(i, valid_names=items) for i in aux_items or []]

    return NetworkItem(x=node_item, values=item, aux=aux_items_str)


def _parse_network_input(
    data: NetworkType,  # Which types should this accept? Res1D would make mikeio1d a dependency...
    name: Optional[str],
    item: str | int | None,
    node_item: str | int | None,
    quantity: Optional[Quantity],
    aux_items: Optional[Sequence[int | str]] = None,
) -> xr.Dataset:
    assert isinstance(
        data, get_args(NetworkType)
    ), "Could not construct NetworkModelResult from provided data."

    data.index.name = "time"
    data = data.reorder_levels(["quantity", "node"], axis=1).stack()
    data = data.reset_index().set_index("time")

    # parse items
    valid_items = list(data.columns)
    sel_items = _parse_network_items(
        valid_items, node_item=node_item, item=item, aux_items=aux_items
    )
    name = name or sel_items.values
    name = _validate_data_var_name(name)

    # parse quantity
    model_quantity = Quantity.undefined() if quantity is None else quantity

    # select relevant items and convert to xr.Dataset
    data = data[sel_items.all]

    ds = data.to_xarray()
    ds = ds.rename({sel_items.node: "node"})
    ds = ds.dropna(dim="time", subset=["node"])
    ds[name].attrs["long_name"] = model_quantity.name
    ds[name].attrs["units"] = model_quantity.unit

    if "node" in ds:
        ds = ds.set_coords(node_item)
    for aux_item in sel_items.aux:
        ds[aux_item].attrs["kind"] = "aux"

    ds.attrs["gtype"] = str(GeometryType.NETWORK)
    assert isinstance(ds, xr.Dataset)
    return ds
