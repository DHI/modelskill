from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from modelskill.timeseries import TimeSeries, _parse_network_node_input
from ._base import SelectedItems
from ..obs import NodeObservation, EdgeObservation
from ..quantity import Quantity
from ..types import PointType

if TYPE_CHECKING:
    from modelskill.network import Network


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
        data: Network,
        *,
        name: str | None = None,
        item: str | int | None = None,
        quantity: Quantity | None = None,
        aux_items: Sequence[int | str] | None = None,
    ):
        self.network = data.copy()

        ds = self.network.to_dataset()
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

    _CHAINAGE_TOLERANCE = 1e-3  # Tolerance in source-network distance units (e.g., meters if chainage is in meters).

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
        observation: NodeObservation | EdgeObservation,
    ) -> NodeModelResult:
        """Extract ModelResult at exact node or edge locations

        Parameters
        ----------
        observation : NodeObservation or EdgeObservation
            observation with node ID or edge ID

        Returns
        -------
        NodeModelResult
            extracted model result
        """
        if isinstance(observation, NodeObservation):
            return self._extract_node(observation)
        elif isinstance(observation, EdgeObservation):
            return self._extract_edge(observation)
        else:
            raise TypeError(
                f"NetworkModelResult supports NodeObservation and EdgeObservation, got {type(observation).__name__}"
            )

    def _extract_node(self, observation: NodeObservation) -> NodeModelResult:
        if observation.at is None and observation.node is None:
            raise ValueError("NodeObservation must have either 'node' or 'at' set")

        raw_id = observation.at if observation.at is not None else observation.node
        assert raw_id is not None  # Redundant assertion, included for mypy
        node_id = self._resolve_alias(raw_id)

        return NodeModelResult(
            data=self.data.sel(node=node_id).drop_vars("node"),
            node=node_id,
            name=self.name,
            item=self.sel_items.values,
            quantity=self.quantity,
            aux_items=self.sel_items.aux,
        )

    def _extract_edge(self, observation: EdgeObservation) -> NodeModelResult:
        # Extract model result from an arbitrary breakpoint belonging to the edge.

        # Searches the alias map for breakpoints whose edge component matches
        # ``observation.edge``, then returns the first one that has data in the
        # dataset.  Raises if no breakpoint with data is found or if the quantity
        # is not present for any breakpoint of that edge.

        item = self.sel_items.values
        edge_id = observation.edge

        try:
            edge = self.network._edges[edge_id]
        except KeyError:
            raise ValueError(f"Edge {edge_id} not found in network.")

        # This only searches intermediate breakpoints since edge-level data is not
        # expected in nodes.

        available_nodes = {int(node_id) for node_id in self.data.node.values}
        found_ds = None
        found_int_id: int | None = None
        missing_node_data = False
        for breakpoint in edge.breakpoints:
            if breakpoint.data is None:
                continue
            if item not in breakpoint.data.columns:
                continue

            int_id = self.network.find(edge=breakpoint.id[0], distance=breakpoint.distance)
            if int_id not in available_nodes:
                missing_node_data = True
                continue

            ds = self.data.sel(node=int_id).drop_vars("node")
            if found_ds is not None:
                da1, da2 = xr.align(ds[item], found_ds[item], join="inner")
                if not np.allclose(da1.values, da2.values, equal_nan=True):
                    raise ValueError(
                        "Not all data in breakpoints are equivalent. "
                        "Select a specific node instead of the edge."
                    )
            else:
                found_ds = ds
                found_int_id = int_id

        if found_ds is not None and found_int_id is not None:
            return NodeModelResult(
                data=found_ds,
                node=found_int_id,
                name=self.name,
                item=item,
                quantity=self.quantity,
                aux_items=self.sel_items.aux,
            )
        if missing_node_data:
            raise ValueError(
                f"Edge '{edge_id}' has breakpoint data for quantity "
                f"'{item}', but matching breakpoint nodes are "
                "missing from the model dataset. Re-create the NetworkModelResult "
                "with the relevant reaches populated."
            )

        raise ValueError(
            f"Edge '{edge_id}' was found in the network but none of its "
            f"breakpoints have data loaded for quantity '{self.sel_items.values}'. "
            f"Re-create the NetworkModelResult with the relevant reaches populated."
        )

    def _resolve_alias(self, alias: int | str | tuple[str, float]) -> int:
        # Resolve a node alias to an internal node ID.

        # Breakpoint tuple aliases are matched first by exact key lookup and then
        # by edge ID and distance within ``_CHAINAGE_TOLERANCE``. If multiple
        # candidates are within tolerance, the closest distance is selected; ties
        # are broken by choosing the smallest node ID. Distance units are the
        # same as the network chainage units.

        if isinstance(alias, int):
            if alias not in self.data.node:
                raise ValueError(
                    f"Node {alias} not found. Available: {list(self.nodes[:5])}..."
                )
            return alias
        else:
            if alias in self.network._alias_map:
                return self.network._alias_map[alias]

            if isinstance(alias, tuple):
                # Handle tolerances
                edge_id, distance = alias
                candidates: list[tuple[float, int]] = []
                for key, node_id in self.network._alias_map.items():
                    if isinstance(key, tuple) and key[0] == edge_id:
                        diff = abs(key[1] - distance)
                        if diff <= self._CHAINAGE_TOLERANCE:
                            candidates.append((diff, node_id))
                if candidates:
                    return min(
                        candidates, key=lambda candidate: (candidate[0], candidate[1])
                    )[1]

            available = list(self.network._alias_map.keys())[:5]
            if isinstance(alias, tuple):
                raise ValueError(
                    f"Breakpoint {alias} not found in network. "
                    f"Available aliases (first 5): {available}"
                )
            raise ValueError(
                f"Node alias '{alias}' not found in network. "
                f"Available aliases (first 5): {available}"
            )
