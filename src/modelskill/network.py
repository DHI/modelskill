"""Opt-in network module for network model results (e.g. MIKE 1D / res1d).

Requires the ``networks`` dependency group (networkx, mikeio1d).
Install with::

    uv sync --group networks

Import this module explicitly to use network functionality::

    from modelskill.network import Network

"""

from __future__ import annotations

import sys

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence, overload, TYPE_CHECKING
from copy import deepcopy

import networkx as nx
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from mikeio1d import Res1D
    from mikeio1d.result_network import ResultReach
    from .model.adapters._res1d import Res1DReach


class NetworkNode(ABC):
    """Abstract base class for a node in a network.

    A node represents a discrete location in the network (e.g. a junction,
    reservoir, or boundary point) that carries time-series data for one or
    more physical quantities.

    Three properties must be implemented:

    * :attr:`id` - a unique string identifier for the node.
    * :attr:`data` - a time-indexed :class:`pandas.DataFrame` whose columns
      are quantity names.
    * :attr:`boundary` - a dict of boundary-condition metadata (may be empty).

    The concrete helper :class:`BasicNode` is provided for the common case
    where the data is already available as a DataFrame.

    See Also
    --------
    BasicNode : Ready-to-use concrete implementation.
    NetworkEdge : Connects two NetworkNode instances.
    Network : Container that assembles nodes and edges into a graph.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique string identifier for this node."""
        pass

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame:
        """Time-indexed DataFrame with one column per quantity."""
        pass

    @property
    @abstractmethod
    def boundary(self) -> dict[str, Any]:
        """Boundary-condition metadata dict (may be empty)."""
        pass

    @property
    def quantities(self) -> list[str]:
        """List of quantity names available at this node."""
        return list(self.data.columns)


class EdgeBreakPoint(ABC):
    """Abstract base class for an intermediate break point along a network edge.

    Break points represent locations between the start and end nodes of an
    edge (e.g. cross-section chainage points along a river reach) that carry
    their own time-series data.

    Two properties must be implemented:

    * :attr:`id` - a ``(edge_id, distance)`` tuple that uniquely locates the
      break point within the network.
    * :attr:`data` - a time-indexed :class:`pandas.DataFrame` whose columns
      are quantity names.

    The :attr:`distance` convenience property returns ``id[1]`` (the
    along-edge distance in the units used by the parent network).

    Examples
    --------
    Minimal subclass:

    >>> class MyBreakPoint(EdgeBreakPoint):
    ...     def __init__(self, edge_id, chainage, df):
    ...         self._id = (edge_id, chainage)
    ...         self._data = df
    ...     @property
    ...     def id(self): return self._id
    ...     @property
    ...     def data(self): return self._data

    See Also
    --------
    NetworkEdge : Owns a list of EdgeBreakPoint instances.
    NetworkNode : Represents a start/end node of an edge.
    Network : Assembles edges (and their break points) into a graph.
    """

    @property
    @abstractmethod
    def id(self) -> tuple[str, float]:
        """``(edge_id, distance)`` tuple uniquely identifying this break point."""
        pass

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame:
        """Time-indexed DataFrame with one column per quantity."""
        pass

    @property
    def distance(self) -> float:
        """Along-edge distance of this break point (same units as :attr:`NetworkEdge.length`)."""
        return self.id[1]

    @property
    def quantities(self) -> list[str]:
        """List of quantity names available at this break point."""
        return list(self.data.columns)


class NetworkEdge(ABC):
    """Abstract base class for an edge in a network.

    An edge represents a directed connection between two :class:`NetworkNode`
    instances (e.g. a river reach between two junctions).  It may also carry
    a list of :class:`EdgeBreakPoint` objects for intermediate chainage
    locations.

    Subclass this to integrate your own network topology.  Five properties
    must be implemented:

    * :attr:`id` - a unique string identifier for the edge.
    * :attr:`start` - the upstream/start :class:`NetworkNode`.
    * :attr:`end` - the downstream/end :class:`NetworkNode`.
    * :attr:`length` - total edge length (in the units of your coordinate
      system).
    * :attr:`breakpoints` - list of :class:`EdgeBreakPoint` instances ordered
      by increasing distance from the start node (empty list if none).

    The concrete helper :class:`BasicEdge` is provided for the common case
    where all data is already available in memory.

    Examples
    --------
    Minimal subclass:

    >>> class MyEdge(NetworkEdge):
    ...     def __init__(self, eid, start_node, end_node, length):
    ...         self._id = eid
    ...         self._start = start_node
    ...         self._end = end_node
    ...         self._length = length
    ...     @property
    ...     def id(self): return self._id
    ...     @property
    ...     def start(self): return self._start
    ...     @property
    ...     def end(self): return self._end
    ...     @property
    ...     def length(self): return self._length
    ...     @property
    ...     def breakpoints(self): return []

    See Also
    --------
    BasicEdge : Ready-to-use concrete implementation.
    NetworkNode : Represents the start/end of this edge.
    EdgeBreakPoint : Intermediate data points along this edge.
    Network : Assembles a list of NetworkEdge objects into a graph.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique string identifier for this edge."""
        pass

    @property
    @abstractmethod
    def start(self) -> NetworkNode:
        """Start (upstream) node of this edge."""
        pass

    @property
    @abstractmethod
    def end(self) -> NetworkNode:
        """End (downstream) node of this edge."""
        pass

    @property
    @abstractmethod
    def length(self) -> float:
        """Total length of this edge in network units."""
        pass

    @property
    @abstractmethod
    def breakpoints(self) -> list[EdgeBreakPoint]:
        """Ordered list of intermediate :class:`EdgeBreakPoint` objects (may be empty)."""
        pass

    @property
    def n_breakpoints(self) -> int:
        """Number of break points in the edge."""
        return len(self.breakpoints)


class BasicNode(NetworkNode):
    """Concrete :class:`NetworkNode` for programmatic network construction.

    Parameters
    ----------
    id : str
        Unique node identifier.
    data : pd.DataFrame
        Time-indexed DataFrame with one column per quantity.
    boundary : dict, optional
        Boundary condition metadata, by default empty.

    Examples
    --------
    >>> import pandas as pd
    >>> time = pd.date_range("2020", periods=3, freq="h")
    >>> node = BasicNode("junction_1", pd.DataFrame({"WaterLevel": [1.0, 1.1, 1.2]}, index=time))
    """

    def __init__(
        self,
        id: str,
        data: pd.DataFrame,
        boundary: dict[str, Any] | None = None,
    ) -> None:
        self._id = id
        self._data = data
        self._boundary: dict[str, Any] = boundary or {}

    @property
    def id(self) -> str:
        return self._id

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def boundary(self) -> dict[str, Any]:
        return self._boundary


class BasicEdge(NetworkEdge):
    """Concrete :class:`NetworkEdge` for programmatic network construction.

    Parameters
    ----------
    id : str
        Unique edge identifier.
    start : NetworkNode
        Start node.
    end : NetworkNode
        End node.
    length : float
        Edge length.
    breakpoints : list[EdgeBreakPoint], optional
        Intermediate break points, by default empty.

    Examples
    --------
    >>> edge = BasicEdge("reach_1", node_a, node_b, length=250.0)
    """

    def __init__(
        self,
        id: str,
        start: NetworkNode,
        end: NetworkNode,
        length: float,
        breakpoints: list[EdgeBreakPoint] | None = None,
    ) -> None:
        self._id = id
        self._start = start
        self._end = end
        self._length = length
        self._breakpoints: list[EdgeBreakPoint] = breakpoints or []

    @property
    def id(self) -> str:
        return self._id

    @property
    def start(self) -> NetworkNode:
        return self._start

    @property
    def end(self) -> NetworkNode:
        return self._end

    @property
    def length(self) -> float:
        return self._length

    @property
    def breakpoints(self) -> list[EdgeBreakPoint]:
        return self._breakpoints


class Network:
    """Network built from a set of edges, with coordinate lookup and data access."""

    def __init__(self, edges: Sequence[NetworkEdge]):
        graph = self._generate_graph(edges)
        self._initialize_network_attributes(graph)
        self._edges = self._generate_edges_dict(edges)

    def _initialize_network_attributes(self, graph: nx.Graph):
        self._alias_map = self._generate_alias_map(graph)
        self._df = self._build_dataframe(graph)
        self._graph = graph.copy()

    def __repr__(self) -> str:
        time = self._df.index
        time_window = "N/A - N/A" if len(time) == 0 else f"{time[0]} - {time[-1]}"
        out = [
            "<Network>",
            f"Edges: {len(self._edges)}",
            f"Nodes: {self._graph.number_of_nodes()}",
            f"Quantities: {self.quantities}",
            f"Time: {time_window}",
        ]
        return "\n".join(out)

    @classmethod
    def from_res1d(
        cls,
        res: str | Path | Res1D,
        *,
        nodes: str | list[str] | None = None,
        reaches: str | list[str] | None = None,
    ) -> Network:
        """Create a Network from a Res1D file or object.

        Parameters
        ----------
        res : str, Path or Res1D
            Path to a .res1d file, or an already-opened :class:`mikeio1d.Res1D` object.
        nodes : str, list of str, or None, optional
            Controls which nodes have their timeseries data loaded into memory.

            * ``None`` *(default)* — data is loaded for every node.
            * A single node ID or a list of node IDs — only those nodes get
              data; others are topology-only.
            * ``[]`` (empty list) — no node data is loaded at all.

            The full network topology is always constructed regardless of this
            setting, so ``find()`` and ``recall()`` still work on all nodes.
        reaches : str, list of str, or None, optional
            Controls which reaches have their intermediate gridpoint data
            populated.

            * ``None`` *(default)* — gridpoints are populated for every reach.
            * A single reach name or a list of reach names — only those reaches
              get gridpoint data; others are topology-only.
            * ``[]`` (empty list) — no gridpoint data is loaded at all.

        Returns
        -------
        Network

        Examples
        --------
        Load everything (default behaviour):

        >>> from modelskill.network import Network
        >>> network = Network.from_res1d("model.res1d")

        Load data only for the two nodes where observations exist, and skip
        all intermediate gridpoint data to keep memory usage low:

        >>> network = Network.from_res1d(
        ...     "model.res1d",
        ...     nodes=["node_a", "node_b"],
        ...     reaches=[],
        ... )

        Load data for selected nodes and gridpoints for one specific reach:

        >>> network = Network.from_res1d(
        ...     "model.res1d",
        ...     nodes=["node_a", "node_b"],
        ...     reaches=["reach_1"],
        ... )
        """

        if sys.version_info >= (3, 14):
            raise NotImplementedError(
                f"Current version of 'mikeio1d' requires python < 3.14 and {sys.version} is being used."
            )

        from mikeio1d import Res1D as _Res1D

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

        if nodes is None:
            nodes_list: list[str] = list(res.nodes.keys())
        elif isinstance(nodes, str):
            nodes_list = [nodes]
        else:
            nodes_list = list(nodes)

        if reaches is None:
            reaches_list: list[str] = list(res.reaches.keys())
        elif isinstance(reaches, str):
            reaches_list = [reaches]
        else:
            reaches_list = list(reaches)

        list_of_reaches = cls._load_res1d_network(res, nodes_list, reaches_list)
        return cls(list_of_reaches)

    @staticmethod
    def _load_res1d_network(
        res: Res1D,
        nodes: list[str],
        reaches: list[str],
    ) -> list[Res1DReach]:
        from modelskill.model.adapters._res1d import (
            Res1DReach,
            Res1DNode,
            _simplify_colnames,
        )

        nodes_set = set(nodes)
        reaches_set = set(reaches)

        # In order to work with bigger files, we might want to select a subset of nodes and avoid
        # potential memory issues. For this reason, we create this intermediate step that populates
        # only the data in the passed nodes

        def _init_node(reach: ResultReach, is_end: bool) -> Res1DNode:
            id = reach.end_node if is_end else reach.start_node
            gpt_idx = -1 if is_end else 0
            if id in nodes_set:
                node = res.nodes[id]
                df = _simplify_colnames(node)
                overlapping_gridpoint = reach.gridpoints[gpt_idx]
                boundary = _simplify_colnames(overlapping_gridpoint)
                return Res1DNode(id, data=df, boundary={reach.name: boundary})
            else:
                return Res1DNode(id)

        return [
            Res1DReach(
                reach,
                _init_node(reach, False),
                _init_node(reach, True),
                populate_gridpoints=reach.name in reaches_set,
            )
            for reach in res.reaches.values()
        ]

    @staticmethod
    def _generate_alias_map(g: nx.Graph) -> dict[str | tuple[str, float], int]:
        return {g.nodes[id]["alias"]: id for id in g.nodes()}

    @staticmethod
    def _generate_edges_dict(edges: Sequence[NetworkEdge]) -> dict[str, NetworkEdge]:
        return {e.id: e for e in edges}

    @staticmethod
    def _build_dataframe(g: nx.Graph) -> pd.DataFrame:
        data_in_nodes = {
            k: v["data"] for k, v in g.nodes.items() if v["data"] is not None
        }
        if len(data_in_nodes) == 0:
            columns = pd.MultiIndex.from_arrays([[], []], names=["node", "quantity"])
            return pd.DataFrame(index=pd.Index([], name="time"), columns=columns)
        df = pd.concat(data_in_nodes, axis=1)
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
        df_raw = self.to_dataframe()
        if len(df_raw.columns) == 0:
            return xr.Dataset()
        df = df_raw.reorder_levels(["quantity", "node"], axis=1)
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

    @staticmethod
    def _generate_graph(edges: Sequence[NetworkEdge]) -> nx.Graph:
        g0 = nx.Graph()
        for edge in edges:
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
    ) -> int:
        pass

    @overload
    def find(
        self,
        *,
        node: list[str],
        edge: None = None,
        distance: None = None,
    ) -> list[int]:
        pass

    @overload
    def find(
        self,
        *,
        node: None = None,
        edge: str | list[str],
        distance: str | float,
    ) -> int:
        pass

    @overload
    def find(
        self,
        *,
        node: None = None,
        edge: str | list[str],
        distance: list[str | float],
    ) -> list[int]:
        pass

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
    def recall(self, id: int) -> dict[str, Any]:
        pass

    @overload
    def recall(self, id: list[int]) -> list[dict[str, Any]]:
        pass

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

    def copy(self) -> "Network":
        """Create a deep copy of the Network.

        Returns
        -------
        Network
            Deep copy of the Network object
        """
        return deepcopy(self)


def _make_basic_network(node_ids, time, data, quantity="WaterLevel"):
    nodes = [
        BasicNode(nid, pd.DataFrame({quantity: data[:, i]}, index=time))
        for i, nid in enumerate(node_ids)
    ]
    edges = [
        BasicEdge(f"e{i}", nodes[i], nodes[i + 1], length=100.0)
        for i in range(len(nodes) - 1)
    ]
    return Network(edges)
