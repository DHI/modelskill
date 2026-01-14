from __future__ import annotations
from typing import Optional, Sequence, Literal

from mikeio1d import Res1D

from ..timeseries import _parse_network_input
from ..quantity import Quantity
from .point import PointModelResult


def build_quantity(network_quantity: str) -> Quantity:
    assert True


class NetworkModelResult(PointModelResult):
    def __init__(
        self,
        data: Res1D | str,
        network_quantity: str,
        *,
        reach: Optional[str] = None,
        node: Optional[int] = None,
        chainage: Optional[float] = None,
        gridpoint: Optional[int | Literal["start", "end"]] = None,
        name: Optional[str] = None,
        quantity: Optional[Quantity] = None,
        aux_items: Optional[Sequence[int | str]] = None,
    ) -> None:
        data = _parse_network_input(
            data,
            quantity=network_quantity,
            reach=reach,
            node=node,
            chainage=chainage,
            gridpoint=gridpoint,
        )
        quantity = build_quantity(network_quantity)
        super().__init__(data=data, name=name, quantity=quantity, aux_items=aux_items)
