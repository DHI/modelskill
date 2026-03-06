"""Opt-in network module for network model results (e.g. MIKE 1D / res1d).

Requires the ``networks`` dependency group (networkx, mikeio1d).
Install with::

    uv sync --group networks

Import this module explicitly to use network functionality::

    from modelskill.network import Network

"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, overload, TYPE_CHECKING

import networkx as nx
import pandas as pd
import xarray as xr

from modelskill.model.network import NetworkEdge, EdgeBreakPoint

if TYPE_CHECKING:
    from mikeio1d import Res1D

# Re-export types users need when building networks
from modelskill.model.network import (  # noqa: F401
    NetworkNode,
    NetworkEdge,
    EdgeBreakPoint,
    BasicNode,
    BasicEdge,
    NodeModelResult,
    NetworkModelResult,
)

__all__ = [
    "Network",
    "NetworkNode",
    "NetworkEdge",
    "EdgeBreakPoint",
    "BasicNode",
    "BasicEdge",
    "NodeModelResult",
    "NetworkModelResult",
]


class Network:
    """Network built from a set of edges, with coordinate lookup and data access."""

    def __init__(self, edges: Sequence[NetworkEdge]):
        self._edges: dict[str, NetworkEdge] = {e.id: e for e in edges}
        self._graph = self._initialize_graph()
        self._alias_map = self._initialize_alias_map()
        self._df = self._build_dataframe()

    @classmethod
    def from_res1d(cls, res: str | Path | Res1D) -> Network:
        """Create a Network from a Res1D file or object.

        Parameters
        ----------
        res : str, Path or Res1D
            Path to a .res1d file, or an already-opened :class:`mikeio1d.Res1D` object.

        Returns
        -------
        Network

        Examples
        --------
        >>> from modelskill.network import Network
        >>> network = Network.from_res1d("model.res1d")
        >>> network = Network.from_res1d(Res1D("model.res1d"))
        """
        from mikeio1d import Res1D as _Res1D
        from modelskill.model.adapters._res1d import Res1DReach

        if isinstance(res, (str, Path)):
            path = Path(res)
            if path.suffix.lower() != ".res1d":
                raise NotImplementedError(
                    f"Unsupported file extension '{path.suffix}'. Only .res1d files are supported."
                )
            res = _Res1D(str(path))
        elif not isinstance(res, _Res1D):
            raise TypeError(
                f"Expected a str, Path or Res1D object, got {type(res).__name__!r}"
            )

        edges = [
            Res1DReach(reach, res.nodes[reach.start_node], res.nodes[reach.end_node])
            for reach in res.reaches.values()
        ]
        return cls(edges)

    def _initialize_alias_map(self) -> dict[str | tuple[str, float], int]:
        return {self.graph.nodes[id]["alias"]: id for id in self.graph.nodes()}

    def _build_dataframe(self) -> pd.DataFrame:
        df = pd.concat({k: v["data"] for k, v in self._graph.nodes.items()}, axis=1)
        df.columns = df.columns.set_names(["node", "quantity"])
        df.index.name = "time"
        return df.copy()

    def to_dataframe(self, sel: str | None = None) -> pd.DataFrame:
        """Dataframe using node ids as column names.

        It will be multiindex unless 'sel' is passed.

        Parameters
        ----------
        sel : Optional[str], optional
            Quantity to select, by default None

        Returns
        -------
        pd.DataFrame
            Timeseries contained in graph nodes
        """
        df = self._df.copy()
        if sel is None:
            return df
        else:
            df.attrs["quantity"] = sel
            return df.reorder_levels(["quantity", "node"], axis=1).loc[:, sel]

    def to_dataset(self) -> xr.Dataset:
        """Dataset using node ids as coords.

        Returns
        -------
        xr.Dataset
            Timeseries contained in graph nodes
        """
        df = self.to_dataframe().reorder_levels(["quantity", "node"], axis=1)
        quantities = df.columns.get_level_values("quantity").unique()
        return xr.Dataset(
            {q: xr.DataArray(df[q], dims=["time", "node"]) for q in quantities}
        )

    @property
    def graph(self) -> nx.Graph:
        """Graph of the network."""
        return self._graph

    @property
    def quantities(self) -> list[str]:
        """Quantities present in data.

        Returns
        -------
        List[str]
            List of quantities
        """
        return list(self.to_dataframe().columns.get_level_values(1).unique())

    def _initialize_graph(self) -> nx.Graph:
        g0 = nx.Graph()
        for edge in self._edges.values():
            # 1) Add start and end nodes
            for node in [edge.start, edge.end]:
                node_key = node.id
                if node_key in g0.nodes:
                    g0.nodes[node_key]["boundary"].update(node.boundary)
                else:
                    g0.add_node(node_key, data=node.data, boundary=node.boundary)

            # 2) Add edges connecting start/end nodes to their adjacent breakpoints
            start_key = edge.start.id
            end_key = edge.end.id
            if edge.n_breakpoints == 0:
                g0.add_edge(start_key, end_key, length=edge.length)
            else:
                bp_keys = [bp.id for bp in edge.breakpoints]
                for bp, bp_key in zip(edge.breakpoints, bp_keys):
                    g0.add_node(bp_key, data=bp.data)

                g0.add_edge(start_key, bp_keys[0], length=edge.breakpoints[0].distance)
                g0.add_edge(
                    bp_keys[-1],
                    end_key,
                    length=edge.length - edge.breakpoints[-1].distance,
                )

            # 3) Connect consecutive intermediate breakpoints
            for i in range(edge.n_breakpoints - 1):
                current_ = edge.breakpoints[i]
                next_ = edge.breakpoints[i + 1]
                length = next_.distance - current_.distance
                g0.add_edge(
                    current_.id,
                    next_.id,
                    length=length,
                )

        return nx.convert_node_labels_to_integers(g0, label_attribute="alias")

    @overload
    def find(
        self,
        *,
        node: str,
        edge: None = None,
        distance: None = None,
    ) -> int: ...

    @overload
    def find(
        self,
        *,
        node: list[str],
        edge: None = None,
        distance: None = None,
    ) -> list[int]: ...

    @overload
    def find(
        self,
        *,
        node: None = None,
        edge: str | list[str],
        distance: str | float,
    ) -> int: ...

    @overload
    def find(
        self,
        *,
        node: None = None,
        edge: str | list[str],
        distance: list[str | float],
    ) -> list[int]: ...

    def find(
        self,
        node: str | list[str] | None = None,
        edge: str | list[str] | None = None,
        distance: str | float | list[str | float] | None = None,
    ) -> int | list[int]:
        """Find node or breakpoint id in the Network object based on former coordinates.

        Parameters
        ----------
        node : str | List[str], optional
            Node id(s) in the original network, by default None
        edge : str | List[str], optional
            Edge id(s) for breakpoint lookup or edge endpoint lookup, by default None
        distance : str | float | List[str | float], optional
            Distance(s) along edge for breakpoint lookup, or "start"/"end"
            for edge endpoints, by default None

        Returns
        -------
        int | List[int]
            Node or breakpoint id(s) in the generic network

        Raises
        ------
        ValueError
            If invalid combination of parameters is provided
        KeyError
            If requested node/breakpoint is not found in the network
        """
        by_node = node is not None
        by_breakpoint = edge is not None or distance is not None

        if by_node and by_breakpoint:
            raise ValueError(
                "Cannot specify both 'node' and 'edge'/'distance' parameters simultaneously"
            )

        if not by_node and not by_breakpoint:
            raise ValueError(
                "Must specify either 'node' or both 'edge' and 'distance' parameters"
            )

        ids: list[str | tuple[str, float]]

        if by_node:
            assert node is not None
            if not isinstance(node, list):
                node = [node]
            ids = list(node)

        else:
            if edge is None or distance is None:
                raise ValueError(
                    "Both 'edge' and 'distance' parameters are required for breakpoint/endpoint lookup"
                )

            if not isinstance(edge, list):
                edge = [edge]

            if not isinstance(distance, list):
                distance = [distance]

            if len(edge) == 1:
                edge = edge * len(distance)

            if len(edge) != len(distance):
                raise ValueError(
                    "Incompatible lengths of 'edge' and 'distance' arguments. One 'edge' admits multiple distances, otherwise they must be the same length."
                )

            ids = []
            for edge_i, distance_i in zip(edge, distance):
                if distance_i in ["start", "end"]:
                    if edge_i not in self._edges:
                        raise KeyError(f"Edge '{edge_i}' not found in the network.")

                    network_edge = self._edges[edge_i]
                    if distance_i == "start":
                        ids.append(network_edge.start.id)
                    else:
                        ids.append(network_edge.end.id)
                else:
                    if not isinstance(distance_i, (int, float)):
                        raise ValueError(
                            "Invalid 'distance' value for breakpoint lookup: "
                            f"{distance_i!r}. Expected a numeric value or 'start'/'end'."
                        )
                    ids.append((edge_i, distance_i))

        _CHAINAGE_TOLERANCE = 1e-3

        def _resolve_id(id):
            if id in self._alias_map:
                return self._alias_map[id]
            if isinstance(id, tuple):
                edge_id, distance = id
                for key, val in self._alias_map.items():
                    if (
                        isinstance(key, tuple)
                        and key[0] == edge_id
                        and abs(key[1] - distance) <= _CHAINAGE_TOLERANCE
                    ):
                        return val
            return None

        resolved = [_resolve_id(id) for id in ids]
        missing_ids = [ids[i] for i, v in enumerate(resolved) if v is None]
        if missing_ids:
            raise KeyError(
                f"Node/breakpoint(s) {missing_ids} not found in the network. Available nodes are {set(self._alias_map.keys())}"
            )
        if len(resolved) == 1:
            return resolved[0]
        return resolved

    @overload
    def recall(self, id: int) -> dict[str, Any]: ...

    @overload
    def recall(self, id: list[int]) -> list[dict[str, Any]]: ...

    def recall(self, id: int | list[int]) -> dict[str, Any] | list[dict[str, Any]]:
        """Recover the original coordinates of an element given the node id(s) in the Network object.

        Parameters
        ----------
        id : int | List[int]
            Node id(s) in the generic network

        Returns
        -------
        Dict[str, Any] | List[Dict[str, Any]]
            Original coordinates. For single input returns dict, for multiple inputs returns list of dicts.
            Dict contains coordinates:
            - For nodes: 'node' key with node id
            - For breakpoints: 'edge' and 'distance' keys with edge id and distance

        Raises
        ------
        KeyError
            If node id is not found in the network
        ValueError
            If node id string format is invalid
        """
        if not isinstance(id, list):
            id = [id]

        reverse_alias_map = {v: k for k, v in self._alias_map.items()}

        results: list[dict[str, Any]] = []
        for node_id in id:
            if node_id not in reverse_alias_map:
                raise KeyError(f"Node ID {node_id} not found in the network.")

            key = reverse_alias_map[node_id]
            if isinstance(key, str):
                results.append({"node": key})
            else:
                results.append({"edge": key[0], "distance": key[1]})

        if len(results) == 1:
            return results[0]
        else:
            return results
