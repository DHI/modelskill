from __future__ import annotations

from typing import Protocol, Sequence, Any, runtime_checkable
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from modelskill.timeseries import TimeSeries, _parse_network_node_input
from ._base import SelectedItems
from ..obs import NodeObservation
from ..quantity import Quantity
from ..types import PointType


@runtime_checkable
class _NetworkLike(Protocol):
    """Duck-type protocol for Network objects passed to NetworkModelResult."""

    def to_dataset(self) -> xr.Dataset: ...


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

    Examples
    --------
    Minimal subclass backed by a CSV file:

    >>> class CsvNode(NetworkNode):
    ...     def __init__(self, node_id, csv_path):
    ...         self._id = node_id
    ...         self._data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    ...     @property
    ...     def id(self): return self._id
    ...     @property
    ...     def data(self): return self._data
    ...     @property
    ...     def boundary(self): return {}

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
        ...

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame:
        """Time-indexed DataFrame with one column per quantity."""
        ...

    @property
    @abstractmethod
    def boundary(self) -> dict[str, Any]:
        """Boundary-condition metadata dict (may be empty)."""
        ...

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
        ...

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame:
        """Time-indexed DataFrame with one column per quantity."""
        ...

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
        ...

    @property
    @abstractmethod
    def start(self) -> NetworkNode:
        """Start (upstream) node of this edge."""
        ...

    @property
    @abstractmethod
    def end(self) -> NetworkNode:
        """End (downstream) node of this edge."""
        ...

    @property
    @abstractmethod
    def length(self) -> float:
        """Total length of this edge in network units."""
        ...

    @property
    @abstractmethod
    def breakpoints(self) -> list[EdgeBreakPoint]:
        """Ordered list of intermediate :class:`EdgeBreakPoint` objects (may be empty)."""
        ...

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

    def _create_new_instance(self, data: xr.Dataset) -> NodeModelResult:
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
        Network-like object with a ``to_dataset()`` method (e.g. :class:`modelskill.network.Network`).
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
    >>> from modelskill.network import Network
    >>> network = Network(edges)  # edges is a list[NetworkEdge]
    >>> mr = ms.NetworkModelResult(network, name="MyModel")
    >>> obs = ms.NodeObservation(data, node=network.find(node="node_A"))
    >>> extracted = mr.extract(obs)
    """

    def __init__(
        self,
        data: _NetworkLike,
        *,
        name: str | None = None,
        item: str | int | None = None,
        quantity: Quantity | None = None,
        aux_items: Sequence[int | str] | None = None,
    ):
        if not isinstance(data, _NetworkLike):
            raise TypeError(
                f"NetworkModelResult expects a Network-like object with 'to_dataset()', got {type(data).__name__!r}"
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
