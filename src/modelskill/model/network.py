from __future__ import annotations
from typing import Sequence
from typing_extensions import Self
import pandas as pd
import xarray as xr

from modelskill.timeseries import TimeSeries, _parse_network_node_input

from ._base import Network1D, SelectedItems
from ..obs import NodeObservation
from ..quantity import Quantity
from ..types import PointType


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
    ) -> None:
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
        node_val = self.data.coords.get("node")
        return int(node_val.item())

    def _create_new_instance(self, data: xr.Dataset) -> Self:
        """Extract node from data and create new instance"""
        node = int(data.coords["node"].item())
        return self.__class__(data, node=node)


class NetworkModelResult(Network1D):
    """Model result for network data with time and node dimensions.

    Construct a NetworkModelResult from an xarray.Dataset with time and node coordinates
    and arbitrary number of data variables. Users must provide exact node IDs (integers)
    when creating observations - no spatial interpolation is performed.

    Parameters
    ----------
    data : xr.Dataset
        xarray.Dataset with time and node coordinates
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
    >>> mr = ms.NetworkModelResult(network_data, name="Network_Model")
    >>> obs = ms.NodeObservation(data, node=123)
    >>> extracted = mr.extract(obs)
    """

    def __init__(
        self,
        data: xr.Dataset,
        *,
        name: str | None = None,
        item: str | int | None = None,
        quantity: Quantity | None = None,
        aux_items: Sequence[int | str] | None = None,
    ) -> None:
        if not isinstance(data, xr.Dataset):
            raise ValueError("'NetworkModelResult' requires xarray.Dataset")
        if len(data.data_vars) == 0:
            raise ValueError("Dataset must have at least one data variable")

        for coord in ["time", "node"]:
            if coord not in data.coords:
                raise ValueError(f"Dataset must have '{coord}' as coordinate.")

        sel_items = SelectedItems.parse(
            list(data.data_vars), item=item, aux_items=aux_items
        )
        name = name or sel_items.values

        self.data: xr.Dataset = data[sel_items.all]
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
                f"Node {node_id} not found. Available: {list(self.data.node.values[:5])}..."
            )

        return NodeModelResult(
            data=self.data.sel(node=node_id).drop_vars("node"),
            node=node_id,
            name=self.name,
            item=self.sel_items.values,
            quantity=self.quantity,
            aux_items=self.sel_items.aux,
        )
