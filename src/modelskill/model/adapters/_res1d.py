from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from mikeio1d.result_network import ResultNode, ResultGridPoint, ResultReach

from modelskill.network import NetworkNode, EdgeBreakPoint, NetworkEdge


def _simplify_colnames(node: ResultNode | ResultGridPoint) -> pd.DataFrame:
    # We remove suffixes and indexes so the columns contain only the quantity names

    # The columns in a Res1D dataframe follow the convention "Quantity:Location:Sublocation"
    # where Location refers to the node id or the reach id followed by the chainage.
    RES1D_NAME_SEP = ":"
    df = node.to_dataframe()
    renamer_dict = {}
    for quantity in node.quantities:
        column_pairs = [(col, quantity) for col in df.columns if quantity in col.split(RES1D_NAME_SEP)]
        if len(column_pairs) != 1:
            raise ValueError(
                f"There must be exactly one column per quantity, found {column_pairs}."
            )
        old_name, new_name = column_pairs[0]
        renamer_dict[old_name] = new_name
    return df.rename(columns=renamer_dict).copy()


class Res1DNode(NetworkNode):
    def __init__(self, id: str, *, data: pd.DataFrame | None = None, boundary: dict[str, pd.DataFrame] = {}):
        self._id = id
        self._data = data
        self._boundary = boundary

    @property
    def id(self) -> str:
        return self._id

    @property
    def data(self) -> pd.DataFrame | None:
        return self._data

    @property
    def boundary(self) -> dict[str, pd.DataFrame]:
        return self._boundary


class GridPoint(EdgeBreakPoint):
    def __init__(self, reach_id: str, chainage: float, data: pd.DataFrame | None = None):
        self._id = (reach_id, chainage)
        self._data = data

    @property
    def id(self) -> tuple[str, float]:
        return self._id

    @property
    def data(self) -> pd.DataFrame | None:
        return self._data


class Res1DReach(NetworkEdge):
    """NetworkEdge adapter for a mikeio1d ResultReach."""

    def __init__(
        self,
        reach: ResultReach,
        start_node: Res1DNode,
        end_node: Res1DNode,
        *,
        populate_gridpoints: bool = True,
    ):
        self._id = reach.name

        if start_node.id != reach.start_node:
            raise ValueError("Incorrect starting node.")
        if end_node.id != reach.end_node:
            raise ValueError("Incorrect ending node.")

        intermediate_gridpoints = (
            reach.gridpoints[1:-1] if len(reach.gridpoints) > 2 else []
        )

        self._start = start_node
        self._end = end_node
        self._length = reach.length
        self._breakpoints: list[EdgeBreakPoint] = [
            GridPoint(
                gridpoint.reach_name,
                gridpoint.chainage,
                _simplify_colnames(gridpoint) if populate_gridpoints else None,
            )
            for gridpoint in intermediate_gridpoints
        ]

    @property
    def id(self) -> str:
        return self._id

    @property
    def start(self) -> Res1DNode:
        return self._start

    @property
    def end(self) -> Res1DNode:
        return self._end

    @property
    def length(self) -> float:
        return self._length

    @property
    def breakpoints(self) -> list[EdgeBreakPoint]:
        return self._breakpoints
