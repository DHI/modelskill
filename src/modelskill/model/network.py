from __future__ import annotations
from typing import Optional, Sequence, Literal
from mikeio1d import Res1D

import pandas as pd

from ..timeseries import _parse_network_input
from ..quantity import Quantity
from .point import PointModelResult
from ..timeseries._coords import NetworkCoords


def read_network_coords(
    data: Res1D, coords: NetworkCoords, variable: Optional[str] = None
) -> pd.DataFrame:
    def variable_name_to_res1d(name: str) -> str:
        return name.replace(" ", "").replace("_", "")

    if ("reaches" not in dir(data)) or ("nodes" not in dir(data)):
        raise ValueError(
            "Invalid file format. Data must have a network structure containing 'nodes' and 'reaches'."
        )

    if coords.by_node and not coords.by_reach:
        location = data.nodes[str(coords.node)]

    if coords.by_reach and not coords.by_node:
        location = data.reaches[coords.reach][coords.chainage]

    df = location.to_dataframe()
    if variable is None:
        if len(df.columns) != 1:
            raise ValueError(
                f"The network location does not have a unique quantity: {location.columns}, in such case 'variable' argument cannot be None"
            )
        return df
    else:
        res1d_name = variable_name_to_res1d(variable)
        relevant_columns = [col for col in df.columns if res1d_name in col]
        assert len(relevant_columns) == 1
        return df.rename(columns={relevant_columns[0]: res1d_name})


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
