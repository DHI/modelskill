from __future__ import annotations
from typing import Optional, Sequence, Literal

from mikeio1d import Res1D

from ..timeseries import _parse_network_input
from ..quantity import Quantity
from .point import PointModelResult


class NetworkModelResult(PointModelResult):
    def __init__(
        self,
        data: Res1D | str,
        quantity: Optional[str | Quantity] = None,
        *,
        reach: Optional[str] = None,
        node: Optional[int] = None,
        chainage: Optional[float] = None,
        gridpoint: Optional[int | Literal["start", "end"]] = None,
        name: Optional[str] = None,
        aux_items: Optional[Sequence[int | str]] = None,
    ) -> None:
        if isinstance(quantity, str):
            quantity = Quantity.from_mikeio_eum_name(quantity)

        data = _parse_network_input(
            data,
            eum_name=quantity.name,
            reach=reach,
            node=node,
            chainage=chainage,
            gridpoint=gridpoint,
        )
        super().__init__(data=data, name=name, quantity=quantity, aux_items=aux_items)
