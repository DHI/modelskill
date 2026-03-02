from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Any, overload
from abc import ABC, abstractmethod
from typing_extensions import Self
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
import networkx as nx

from modelskill.timeseries import TimeSeries, _parse_network_node_input

if TYPE_CHECKING:
    from mikeio1d import Res1D

from ._base import SelectedItems
from ..obs import NodeObservation
from ..quantity import Quantity
from ..types import PointType


class NetworkNode(ABC):
    """Node in the simplified network."""

    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame: ...

    @property
    @abstractmethod
    def boundary(self) -> dict[str, Any]: ...

    @property
    def quantities(self) -> list[str]:
        return list(self.data.columns)


class EdgeBreakPoint(ABC):
    """Edge break point."""

    @property
    @abstractmethod
    def id(self) -> tuple[str, float]: ...

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame: ...

    @property
    def distance(self) -> float:
        return self.id[1]

    @property
    def quantities(self) -> list[str]:
        return list(self.data.columns)


class NetworkEdge(ABC):
    """Edge of a network."""

    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def start(self) -> NetworkNode: ...

    @property
    @abstractmethod
    def end(self) -> NetworkNode: ...

    @property
    @abstractmethod
    def length(self) -> float: ...

    @property
    @abstractmethod
    def breakpoints(self) -> list[EdgeBreakPoint]: ...

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

    def __init__(self, edges: list[NetworkEdge]):
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
        >>> from modelskill.model.network import Network
        >>> network = Network.from_res1d("model.res1d")
        >>> network = Network.from_res1d(Res1D("model.res1d"))
        """
        from mikeio1d import Res1D as _Res1D
        from .adapters._res1d import Res1DReach

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
            Res1DReach(
                reach, res.nodes[reach.start_node], res.nodes[reach.end_node]
            )
            for reach in res.reaches.values()
        ]
        return cls(edges)

    def _initialize_alias_map(self)-> dict[str | tuple[str, float], int]:
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
        """Find node or breakpoint id in the generic network.

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
        # Determine lookup mode
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

        if by_node:
            # Handle node lookup
            if not isinstance(node, list):
                node = [node]
            ids = list(node)

        else:
            # Handle breakpoint/edge endpoint lookup
            if edge is None or distance is None:
                raise ValueError(
                    "Both 'edge' and 'distance' parameters are required for breakpoint/endpoint lookup"
                )

            if not isinstance(edge, list):
                edge = [edge]

            if not isinstance(distance, list):
                distance = [distance]

            # We can pass one edge and multiple breakpoints/endpoints
            if len(edge) == 1:
                edge = edge * len(distance)

            if len(edge) != len(distance):
                raise ValueError(
                    "Incompatible lengths of 'edge' and 'distance' arguments. One 'edge' admits multiple distances, otherwise they must be the same length."
                )

            ids = []
            for edge_i, distance_i in zip(edge, distance):
                if distance_i in ["start", "end"]:
                    # Handle edge endpoint lookup
                    if edge_i not in self._edges:
                        raise KeyError(f"Edge '{edge_i}' not found in the network.")

                    network_edge = self._edges[edge_i]
                    if distance_i == "start":
                        ids.append(network_edge.start.id)
                    else:  # distance_i == "end"
                        ids.append(network_edge.end.id)
                else:
                    # Handle breakpoint lookup
                    if not isinstance(distance_i, (int, float)):
                        raise ValueError(
                            "Invalid 'distance' value for breakpoint lookup: "
                            f"{distance_i!r}. Expected a numeric value or 'start'/'end'."
                        )
                    ids.append((edge_i, distance_i))

        # Check if all ids exist in the network
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
        """Recall the original coordinates from generic network node id(s).

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
        # Convert to list for uniform processing
        if not isinstance(id, list):
            id = [id]

        # Create reverse lookup map
        reverse_alias_map = {v: k for k, v in self._alias_map.items()}

        results = []
        for node_id in id:
            if node_id not in reverse_alias_map:
                raise KeyError(f"Node ID {node_id} not found in the network.")

            key = reverse_alias_map[node_id]
            if isinstance(key, str):
                results.append({"node": key})
            else:  # tuple[str, float]
                results.append({"edge": key[0], "distance": key[1]})

        # Return single dict if single input, list otherwise
        if len(results) == 1:
            return results[0]
        else:
            return results


class NodeModelResult(TimeSeries):
    """Model result for a single network node.

    Construct a NodeModelResult from timeseries data for a specific node.
    This is a simple timeseries class designed for network node data.

    Parameters
    ----------
    data : str, Path, mikeio.Dataset, mikeio.DataArray, pd.DataFrame, pd.Series, xr.Dataset or xr.DataArray
        filename (.dfs0 or .nc) or object with the data
    name : str, optional
        The name of the model result,
        by default None (will be set to file name or item name)
    node : int, optional
        node ID (integer), by default None
    item : str | int | None, optional
        If multiple items/arrays are present in the input an item
        must be given (as either an index or a string), by default None
    quantity : Quantity, optional
        Model quantity, for MIKE files this is inferred from the EUM information
    aux_items : list[int | str], optional
        Auxiliary items, by default None

    Examples
    --------
    >>> import modelskill as ms
    >>> mr = ms.NodeModelResult(data, node=123, name="Node_123")
    >>> mr2 = ms.NodeModelResult(df, item="Water Level", node=456)
    """

    def __init__(
        self,
        data: PointType,
        node: int,
        *,
        name: str | None = None,
        item: str | int | None = None,
        quantity: Quantity | None = None,
        aux_items: Sequence[int | str] | None = None,
    ):
        if not self._is_input_validated(data):
            data = _parse_network_node_input(
                data,
                name=name,
                item=item,
                quantity=quantity,
                node=node,
                aux_items=aux_items,
            )

        if not isinstance(data, xr.Dataset):
            raise ValueError("'NodeModelResult' requires xarray.Dataset")
        if data.coords.get("node") is None:
            raise ValueError("'node' coordinate not found in data")
        data_var = str(list(data.data_vars)[0])
        data[data_var].attrs["kind"] = "model"
        super().__init__(data=data)

    @property
    def node(self) -> int:
        """Node ID of model result"""
        node_val = self.data.coords["node"]
        return int(node_val.item())

    def _create_new_instance(self, data: xr.Dataset) -> Self:
        """Extract node from data and create new instance"""
        node = int(data.coords["node"].item())
        return self.__class__(data, node=node)


class NetworkModelResult:
    """Model result for network data with time and node dimensions.

    Construct a NetworkModelResult from a Network object containing
    timeseries data for each node. Users must provide exact node IDs
    (integers obtained via ``Network.find()``) when creating observations —
    no spatial interpolation is performed.

    Parameters
    ----------
    data : Network
        Network object containing timeseries data for each node.
    name : str, optional
        The name of the model result,
        by default None (will be set to first data variable name)
    item : str | int | None, optional
        If multiple items/arrays are present in the input an item
        must be given (as either an index or a string), by default None
    quantity : Quantity, optional
        Model quantity
    aux_items : list[int | str], optional
        Auxiliary items, by default None

    Examples
    --------
    >>> import modelskill as ms
    >>> from modelskill.model.network import Network
    >>> network = Network(edges)  # edges is a list[NetworkEdge]
    >>> mr = ms.NetworkModelResult(network, name="MyModel")
    >>> obs = ms.NodeObservation(data, node=network.find(node="node_A"))
    >>> extracted = mr.extract(obs)
    """

    def __init__(
        self,
        data: Network,
        *,
        name: str | None = None,
        item: str | int | None = None,
        quantity: Quantity | None = None,
        aux_items: Sequence[int | str] | None = None,
    ):
        if not isinstance(data, Network):
            raise TypeError(
                f"NetworkModelResult expects a Network object, got {type(data).__name__!r}"
            )
        ds = data.to_dataset()
        sel_items = SelectedItems.parse(
            list(ds.data_vars), item=item, aux_items=aux_items
        )
        name = name or sel_items.values

        self.data = ds[sel_items.all]
        self.name = name
        self.sel_items = sel_items

        if quantity is None:
            da = self.data[sel_items.values]
            quantity = Quantity.from_cf_attrs(da.attrs)
        self.quantity = quantity

        # Mark data variables as model data
        self.data[sel_items.values].attrs["kind"] = "model"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>: {self.name}"

    @property
    def time(self) -> pd.DatetimeIndex:
        """Return the time coordinate as a pandas.DatetimeIndex."""
        return pd.DatetimeIndex(self.data.time.to_index())

    @property
    def nodes(self) -> npt.NDArray[np.intp]:
        """Return the node IDs as a numpy array of integers."""
        return self.data.node.values

    def extract(
        self,
        observation: NodeObservation,
    ) -> NodeModelResult:
        """Extract ModelResult at exact node locations

        Parameters
        ----------
        observation : NodeObservation
            observation with node ID (only NodeObservation supported)

        Returns
        -------
        NodeModelResult
            extracted model result
        """
        if not isinstance(observation, NodeObservation):
            raise TypeError(
                f"NetworkModelResult only supports NodeObservation, got {type(observation).__name__}"
            )

        node_id = observation.node
        if node_id not in self.data.node:
            raise ValueError(
                f"Node {node_id} not found. Available: {list(self.nodes[:5])}..."
            )

        return NodeModelResult(
            data=self.data.sel(node=node_id).drop_vars("node"),
            node=node_id,
            name=self.name,
            item=self.sel_items.values,
            quantity=self.quantity,
            aux_items=self.sel_items.aux,
        )
