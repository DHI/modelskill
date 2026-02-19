from __future__ import annotations
from typing import Any, Optional, Sequence
import numpy as np

import xarray as xr
import pandas as pd

from modelskill.obs import Observation
from modelskill.timeseries import TimeSeries, _parse_network_node_input

from ._base import SpatialField, _validate_overlap_in_time, SelectedItems
from ..obs import NodeObservation
from ..quantity import Quantity
from ..types import PointType


class NodeModelResult(TimeSeries):
    """Model result for a single network node.

    Construct a NodeModelResult from timeseries data for a specific node ID.
    This is a simple timeseries class designed for network node data.

    Parameters
    ----------
    data : str, Path, mikeio.Dataset, mikeio.DataArray, pd.DataFrame, pd.Series, xr.Dataset or xr.DataArray
        filename (.dfs0 or .nc) or object with the data
    name : Optional[str], optional
        The name of the model result,
        by default None (will be set to file name or item name)
    node : int, optional
        node ID (integer), by default None
    item : str | int | None, optional
        If multiple items/arrays are present in the input an item
        must be given (as either an index or a string), by default None
    quantity : Quantity, optional
        Model quantity, for MIKE files this is inferred from the EUM information
    aux_items : Optional[list[int | str]], optional
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
        *,
        name: Optional[str] = None,
        node: Optional[int] = None,
        item: str | int | None = None,
        quantity: Optional[Quantity] = None,
        aux_items: Optional[Sequence[int | str]] = None,
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

        assert isinstance(data, xr.Dataset)

        data_var = str(list(data.data_vars)[0])
        data[data_var].attrs["kind"] = "model"
        super().__init__(data=data)

    @property
    def node(self) -> int:
        """Node ID of model result"""
        node_val = self.data.coords.get("node")
        if node_val is not None:
            return int(node_val.item())
        return None

    def interp_time(self, observation: Observation, **kwargs: Any) -> NodeModelResult:
        """
        Interpolate model result to the time of the observation

        wrapper around xarray.Dataset.interp()

        Parameters
        ----------
        observation : Observation
            The observation to interpolate to
        **kwargs
            Additional keyword arguments passed to xarray.interp

        Returns
        -------
        NodeModelResult
            Interpolated model result
        """
        ds = self.align(observation, **kwargs)
        return NodeModelResult(ds)

    def align(
        self,
        observation: Observation,
        *,
        max_gap: float | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        new_time = observation.time

        dati = self.data.dropna("time").interp(
            time=new_time, assume_sorted=True, **kwargs
        )

        nmr = NodeModelResult(dati)
        if max_gap is not None:
            nmr = nmr._remove_model_gaps(mod_index=self.time, max_gap=max_gap)
        return nmr.data

    def _remove_model_gaps(
        self,
        mod_index: pd.DatetimeIndex,
        max_gap: float | None = None,
    ) -> NodeModelResult:
        """Remove model gaps longer than max_gap from TimeSeries"""
        max_gap_delta = pd.Timedelta(max_gap, "s")
        valid_times = self._get_valid_times(mod_index, max_gap_delta)
        ds = self.data.sel(time=valid_times)
        return NodeModelResult(ds)

    def _get_valid_times(
        self, mod_index: pd.DatetimeIndex, max_gap: pd.Timedelta
    ) -> pd.DatetimeIndex:
        """Used only by _remove_model_gaps"""
        obs_index = self.time
        # init dataframe of available timesteps and their index
        df = pd.DataFrame(index=mod_index)
        df["idx"] = range(len(df))

        # for query times get available left and right index of source times
        df = (
            df.reindex(df.index.union(obs_index))
            .interpolate(method="time", limit_area="inside")
            .reindex(obs_index)
            .dropna()
        )
        df["idxa"] = np.floor(df.idx).astype(int)
        df["idxb"] = np.ceil(df.idx).astype(int)

        # time of left and right source times and time delta
        df["ta"] = mod_index[df.idxa]
        df["tb"] = mod_index[df.idxb]
        df["dt"] = df.tb - df.ta

        # valid query times where time delta is less than max_gap
        valid_idx = df.dt <= max_gap
        return df[valid_idx].index


class NetworkModelResult(SpatialField):
    """Model result for network data with time and node dimensions.

    Construct a NetworkModelResult from an xarray.Dataset with time and node coordinates
    and arbitrary number of data variables. Users must provide exact node IDs (integers)
    when creating observations - no spatial interpolation is performed.

    Parameters
    ----------
    data : xr.Dataset
        xarray.Dataset with time and node coordinates
    name : Optional[str], optional
        The name of the model result,
        by default None (will be set to first data variable name)
    item : str | int | None, optional
        If multiple items/arrays are present in the input an item
        must be given (as either an index or a string), by default None
    quantity : Quantity, optional
        Model quantity
    aux_items : Optional[list[int | str]], optional
        Auxiliary items, by default None

    Notes
    -----
    For point observations, specify the node ID as the x-coordinate (integer).
    No spatial interpolation between nodes is performed.
    """

    def __init__(
        self,
        data: xr.Dataset,
        *,
        name: Optional[str] = None,
        item: str | int | None = None,
        quantity: Optional[Quantity] = None,
        aux_items: Optional[Sequence[int | str]] = None,
    ) -> None:
        assert isinstance(
            data, xr.Dataset
        ), "NetworkModelResult requires xarray.Dataset"
        assert "time" in data.dims, "Dataset must have time dimension"
        assert "node" in data.dims, "Dataset must have node dimension"
        assert len(data.data_vars) > 0, "Dataset must have at least one data variable"

        sel_items = SelectedItems.parse(
            list(data.data_vars), item=item, aux_items=aux_items
        )
        name = name or sel_items.values

        self.data: xr.Dataset = data[sel_items.all]
        self.name = name
        self.sel_items = sel_items

        # use long_name and units from data if not provided
        if quantity is None:
            da = self.data[sel_items.values]
            quantity = Quantity.from_cf_attrs(da.attrs)

        self.quantity = quantity

        # Mark data variables as model data
        data_var = sel_items.values
        self.data[data_var].attrs["kind"] = "model"

    def __repr__(self) -> str:
        res = []
        res.append(f"<{self.__class__.__name__}>: {self.name}")
        res.append(f"Time: {self.time[0]} - {self.time[-1]}")
        res.append(f"Nodes: {len(self.data.node)} nodes")
        res.append(f"Quantity: {self.quantity}")
        if len(self.sel_items.aux) > 0:
            res.append(f"Auxiliary variables: {', '.join(self.sel_items.aux)}")
        return "\n".join(res)

    @property
    def time(self) -> pd.DatetimeIndex:
        return self.data.time

    def extract(
        self,
        observation: NodeObservation,
    ) -> NodeModelResult:
        """Extract ModelResult at exact node locations

        Note: this method is typically not called directly, but through the match() method.
        The observation must specify the exact node ID.

        Parameters
        ----------
        observation : <NodeObservation>
            observation with node ID

        Returns
        -------
        NodeModelResult
            extracted modelresult
        """
        _validate_overlap_in_time(self.time, observation)
        if isinstance(observation, NodeObservation):
            return self._extract_node(observation)
        else:
            raise NotImplementedError(
                f"NetworkModelResult only supports NodeObservation extraction, not {type(observation).__name__}."
            )

    def _extract_node(self, observation: NodeObservation) -> NodeModelResult:
        """Extract node data from network.

        The observation specifies the exact node ID.
        No spatial interpolation is performed - the exact node is selected.
        """
        node_id = observation.node
        if node_id is None:
            raise ValueError(
                f"NodeObservation '{observation.name}' must specify a valid node ID."
            )

        # Check if node exists in the network
        if node_id not in self.data.node.values:
            available_nodes = list(self.data.node.values)
            raise ValueError(
                f"Node ID {node_id} not found in network. Available nodes: {available_nodes[:10]}{'...' if len(available_nodes) > 10 else ''}"
            )

        # Extract data at the specified node
        ds = self.data.sel(node=node_id)

        # Convert to dataframe and create NodeModelResult
        df = ds.to_dataframe().dropna()
        if len(df) == 0:
            raise ValueError(
                f"No data available for node {node_id} in NetworkModelResult '{self.name}'"
            )
        df = df.rename(columns={self.sel_items.values: self.name})

        return NodeModelResult(
            data=df,
            node=node_id,
            item=self.name,
            name=self.name,
            quantity=self.quantity,
            aux_items=self.sel_items.aux,
        )
