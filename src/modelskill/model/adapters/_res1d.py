from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from mikeio1d.result_network import ResultNode, ResultGridPoint, ResultReach

from modelskill.network import NetworkNode, ReachBreakPoint, NetworkReach


def _simplify_colnames(node: ResultNode | ResultGridPoint) -> pd.DataFrame:
    # We remove suffixes and indexes so the columns contain only the quantity names

    # Some formats keep no timeseries at all on some locations - MIKE 11, for instance,
    # stores everything on reach gridpoints, leaving the nodes empty. Asking mikeio1d
    # for a dataframe there raises, so return an empty one instead.
    if not node.quantities:
        return pd.DataFrame()

    # The columns in a Res1D dataframe follow the convention "Quantity:Location:Sublocation"
    # where Location refers to the node id or the reach id followed by the chainage.
    RES1D_NAME_SEP = ":"
    df = node.to_dataframe()
    renamer_dict = {}
    for quantity in node.quantities:
        column_pairs = [
            (col, quantity)
            for col in df.columns
            if quantity in col.split(RES1D_NAME_SEP)
        ]
        if len(column_pairs) != 1:
            raise ValueError(
                f"There must be exactly one column per quantity, found {column_pairs}."
            )
        old_name, new_name = column_pairs[0]
        renamer_dict[old_name] = new_name
    return df.rename(columns=renamer_dict).copy()


def _merge_extra_quantities(
    base: pd.DataFrame, extra: pd.DataFrame, *, node_id: str
) -> pd.DataFrame:
    """Append a companion file's quantities to a node's frame as extra columns.

    Parameters
    ----------
    base : pd.DataFrame
        The node's frame from the main result file.
    extra : pd.DataFrame
        The same node's frame from the companion file, sharing its time index.
    node_id : str
        Node ID, used in error messages.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If a quantity appears in both frames. Concatenating would give the node
        two columns of the same name, which is the state ``_simplify_colnames``
        already refuses.
    """
    if extra.empty:
        return base

    overlapping = base.columns.intersection(extra.columns)
    if len(overlapping) > 0:
        raise ValueError(
            f"Node {node_id!r} already has {sorted(overlapping)} in the main "
            "result file, so the companion file's copy cannot be merged in."
        )

    return pd.concat([base, extra], axis=1)


class Res1DNode(NetworkNode):
    def __init__(
        self,
        id: str,
        *,
        data: pd.DataFrame | None = None,
        boundary: dict[str, pd.DataFrame] | None = None,
    ):
        self._id = id
        self._data = pd.DataFrame() if data is None else data
        self._boundary = {} if boundary is None else boundary

    @property
    def id(self) -> str:
        return self._id

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def boundary(self) -> dict[str, pd.DataFrame]:
        return self._boundary


class GridPoint(ReachBreakPoint):
    def __init__(
        self, reach_id: str, chainage: float, data: pd.DataFrame | None = None
    ):
        self._id = (reach_id, chainage)
        self._data = pd.DataFrame() if data is None else data

    @property
    def id(self) -> tuple[str, float]:
        return self._id

    @property
    def data(self) -> pd.DataFrame:
        return self._data


class Res1DReach(NetworkReach):
    """NetworkReach adapter for a mikeio1d ResultReach."""

    def __init__(
        self,
        reach: ResultReach,
        start_node: Res1DNode,
        end_node: Res1DNode,
        *,
        populate_gridpoints: bool = True,
        length: float | None = None,
    ):
        self._id = reach.name

        # Must be checked separately: some formats (.resx) report None for both the
        # reach and the node, which the identity checks below would let through.
        if reach.start_node is None or reach.end_node is None:
            raise ValueError(
                f"mikeio1d reported no start/end node for reach {reach.name!r}; "
                "this result format's topology cannot be represented as a Network."
            )

        if start_node.id != reach.start_node:
            raise ValueError("Incorrect starting node.")
        if end_node.id != reach.end_node:
            raise ValueError("Incorrect ending node.")

        intermediate_gridpoints = (
            reach.gridpoints[1:-1] if len(reach.gridpoints) > 2 else []
        )

        self._start = start_node
        self._end = end_node

        # A length read from a companion input file wins, since mikeio1d has none
        # to offer for the formats that need one. Otherwise: mikeio1d returns 0
        # when it cannot read a reach length - link-node models such as EPANET
        # report this for every reach. Report it as undefined rather than as a
        # zero-length reach, which would make length-weighted graph algorithms
        # treat the reach as free. The two cases cannot be told apart upstream.
        self._length = length if length is not None else (reach.length or None)
        self._breakpoints: list[ReachBreakPoint] = [
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
    def length(self) -> float | None:
        return self._length

    @property
    def breakpoints(self) -> list[ReachBreakPoint]:
        return self._breakpoints
