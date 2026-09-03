from __future__ import annotations
from typing import Sequence

import xarray as xr

from ..quantity import Quantity
from ..types import PointType
from ._coords import NodeCoords, ReachCoords
from ._point import _parse_point_input


def _parse_network_node_input(
    data: PointType,
    name: str | None,
    item: str | int | None,
    quantity: Quantity | None,
    node: str | None,
    aux_items: Sequence[int | str] | None,
) -> xr.Dataset:
    if node is None:
        raise ValueError("'node' argument cannot be empty.")
    coords = NodeCoords(node=node)
    ds = _parse_point_input(data, name, item, quantity, aux_items, coords=coords)
    return ds


def _parse_network_breakpoint_input(
    data: PointType,
    name: str | None,
    item: str | int | None,
    quantity: Quantity | None,
    aux_items: Sequence[int | str] | None,
    *,
    reach: str,
    distance: float | None = None,
) -> xr.Dataset:
    """Parse input for a breakpoint (or reach-level) observation.

    When ``distance`` is ``None`` the observation is reach-level — no
    ``distance`` coordinate is stored and the result can be matched to any
    breakpoint on the reach.
    """
    coords = ReachCoords(reach=reach, distance=distance)
    ds = _parse_point_input(data, name, item, quantity, aux_items, coords=coords)
    return ds
