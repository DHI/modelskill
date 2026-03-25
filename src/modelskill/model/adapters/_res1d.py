from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from mikeio1d.result_network import ResultNode, ResultGridPoint, ResultReach

from modelskill.network import NetworkNode, EdgeBreakPoint, NetworkEdge


def _simplify_res1d_colnames(node: ResultNode | ResultGridPoint) -> pd.DataFrame:
    # We remove suffixes and indexes so the columns contain only the quantity names
    RES1D_NAME_SEP = ":"
    df = node.to_dataframe()
    renamer_dict = {}
    for quantity in node.quantities:
        column_pairs = [(col, quantity) for col in df.columns if quantity in col.split(RES1D_NAME_SEP)]
        if len(column_pairs) != 1:
            raise ValueError(
                f"There must be exactly one column per quantity, found {column_pairs}."
            )
        old_name, new_name = column_pairs
        renamer_dict[old_name] = new_name
    return df.rename(columns=renamer_dict).copy()


class Res1DNode(NetworkNode):
    def __init__(self, node: ResultNode, boundary: dict[str, ResultGridPoint]):
        self._id = node.id
        self._data = _simplify_res1d_colnames(node)
        self._boundary = {
            key: _simplify_res1d_colnames(point) for key, point in boundary.items()
        }

    @property
    def id(self) -> str:
        return self._id

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def boundary(self) -> dict[str, pd.DataFrame]:
        return self._boundary


class GridPoint(EdgeBreakPoint):
    def __init__(self, point: ResultGridPoint):
        self._id = (point.reach_name, point.chainage)
        self._data = _simplify_res1d_colnames(point)

    @property
    def id(self) -> tuple[str, float]:
        return self._id

    @property
    def data(self) -> pd.DataFrame:
        return self._data


class Res1DReach(NetworkEdge):
    """NetworkEdge adapter for a mikeio1d ResultReach."""

    def __init__(
        self, reach: ResultReach, start_node: ResultNode, end_node: ResultNode
    ):
        self._id = reach.name

        if start_node.id != reach.start_node:
            raise ValueError("Incorrect starting node.")
        if end_node.id != reach.end_node:
            raise ValueError("Incorrect ending node.")

        start_gridpoint = reach.gridpoints[0]
        end_gridpoint = reach.gridpoints[-1]
        intermediate_gridpoints = (
            reach.gridpoints[1:-1] if len(reach.gridpoints) > 2 else []
        )

        self._start = Res1DNode(start_node, {reach.name: start_gridpoint})
        self._end = Res1DNode(end_node, {reach.name: end_gridpoint})
        self._length = reach.length
        self._breakpoints: list[EdgeBreakPoint] = [
            GridPoint(gridpoint) for gridpoint in intermediate_gridpoints
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
