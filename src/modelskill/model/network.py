from __future__ import annotations
from typing import Optional, Sequence, Literal

import pandas as pd

from pathlib import Path
from mikeio1d import Res1D, open

from ..quantity import Quantity
from .point import PointModelResult


def parse_network_location(
    data: Res1D | str,
    quantity: str,
    *,
    node: Optional[int] = None,
    reach: Optional[str] = None,
    chainage: Optional[str | float] = None,
    gridpoint: Optional[int | Literal["start", "end"]] = None,
) -> pd.Series:
    if isinstance(data, (str, Path)):
        if Path(data).suffix == ".res1d":
            data = open(data)
        else:
            raise ValueError("Invalid path to network")

    by_node = node is not None
    by_reach = reach is not None
    with_chainage = chainage is not None
    with_index = gridpoint is not None

    if by_node and not by_reach:
        location = data.nodes[str(node)]
        assert (
            quantity in location.quantities
        ), f"Quantity {quantity} was not found in node."

    elif by_reach and not by_node:
        location = data.reaches[reach]
        if with_index != with_chainage:
            raise ValueError(
                "Items accessed by chainage must be specified either by chainage or by index, not both"
            )

        if with_index and not with_chainage:
            gridpoint = 0 if gridpoint == "start" else gridpoint
            gridpoint = -1 if gridpoint == "end" else gridpoint
            chainage = location.chainages[gridpoint]

        location = location[chainage]

    else:
        raise ValueError("Item can only be specified either by node or by reach")

    df = location.to_dataframe()
    assert df.shape[1] == 1, "Multiple columns found in df"
    df.rename(columns=lambda x: quantity, inplace=True)
    return df[quantity].copy()


class NetworkModelResult(PointModelResult):
    def __init__(
        self,
        data: Res1D | str,
        network_quantity: str,
        *,
        reach: Optional[float] = None,
        node: Optional[float] = None,
        chainage: Optional[float] = None,
        gridpoint: Optional[int | Literal["start", "end"]] = None,
        name: Optional[str] = None,
        quantity: Optional[Quantity] = None,
        aux_items: Optional[Sequence[int | str]] = None,
    ) -> None:
        data = parse_network_location(
            data,
            quantity=network_quantity,
            reach=reach,
            node=node,
            chainage=chainage,
            gridpoint=gridpoint,
        )
        super().__init__(data=data, name=name, quantity=quantity, aux_items=aux_items)
