from __future__ import annotations
from typing import Optional, Sequence, Literal
from mikeio1d import Res1D


from ..timeseries import _parse_network_input
from ..quantity import Quantity
from .point import PointModelResult


class NetworkModelResult(PointModelResult):
    """Model result for a network location.

    Construct a NetworkModelResult from a res1d data source.

    Parameters
    ----------
    data : str, Path or mikeio1d.Res1D
        filename (.res1d) or object with the data
    quantity : str
        The name of the model result,
        by default None (will be set to file name or item name)
    reach : str, optional
        Reach id in the network
    node : int, optional
        Node id in the network
    chainage : float, optional
        Chainage number in its respective reach
    gridpoint : int, optional
        Index associated to the gridpoints in the reach
    name : Optional[str], optional
        The name of the model result,
        by default None (will be set to file name or item name)
    aux_items : Optional[list[int | str]], optional
        Auxiliary items, by default None
    """

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

        variable = quantity.name if isinstance(quantity, Quantity) else None
        data = _parse_network_input(
            data,
            variable=variable,
            reach=reach,
            node=node,
            chainage=chainage,
            gridpoint=gridpoint,
        )
        super().__init__(data=data, name=name, quantity=quantity, aux_items=aux_items)
