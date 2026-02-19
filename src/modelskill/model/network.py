from __future__ import annotations
from typing import Optional, Sequence
import numpy as np

import xarray as xr
import pandas as pd

from ._base import SpatialField, _validate_overlap_in_time, SelectedItems
from ..obs import PointObservation
from ..quantity import Quantity
from .point import PointModelResult


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
        return pd.DatetimeIndex(self.data.time)

    def extract(
        self,
        observation: PointObservation,
        spatial_method: Optional[str] = None,
    ) -> PointModelResult:
        """Extract ModelResult at exact node locations

        Note: this method is typically not called directly, but through the match() method.
        The observation's x-coordinate must specify the exact node ID (integer).

        Parameters
        ----------
        observation : <PointObservation>
            observation where x-coordinate specifies the node ID (as integer)
        spatial_method : Optional[str], optional
            Not used for network extraction (exact node selection)

        Returns
        -------
        PointModelResult
            extracted modelresult
        """
        _validate_overlap_in_time(self.time, observation)
        if isinstance(observation, PointObservation):
            return self._extract_point(observation, spatial_method)
        else:
            raise NotImplementedError(
                f"NetworkModelResult only supports PointObservation extraction, not {type(observation).__name__}."
            )

    def _extract_point(
        self, observation: PointObservation, spatial_method: Optional[str] = None
    ) -> PointModelResult:
        """Extract point from network data using exact node ID.

        The observation's x-coordinate should specify the node ID (as integer).
        No spatial interpolation is performed - the exact node is selected.
        """
        node_id = observation.x
        if node_id is None:
            raise ValueError(
                f"PointObservation '{observation.name}' must specify node ID in x-coordinate."
            )

        # Convert to integer node ID
        try:
            node_id = int(node_id)
        except (ValueError, TypeError):
            raise ValueError(
                f"Node ID (x-coordinate) must be an integer, got {node_id} for observation '{observation.name}'"
            )

        # Check if node exists in the network
        if node_id not in self.data.node.values:
            available_nodes = list(self.data.node.values)
            raise ValueError(
                f"Node ID {node_id} not found in network. Available nodes: {available_nodes[:10]}{'...' if len(available_nodes) > 10 else ''}"
            )

        # Extract data at the specified node
        ds = self.data.sel(node=node_id)

        # Convert to dataframe and create PointModelResult
        df = ds.to_dataframe().dropna()
        if len(df) == 0:
            raise ValueError(
                f"No data available for node {node_id} in NetworkModelResult '{self.name}'"
            )
        df = df.rename(columns={self.sel_items.values: self.name})

        # Use observation coordinates or extract from data if available
        x_coord = observation.x if observation.x is not None else node_id
        y_coord = (
            observation.y
            if observation.y is not None
            else (float(ds.y.item()) if hasattr(ds, "y") and ds.y.size == 1 else np.nan)
        )

        return PointModelResult(
            data=df,
            x=x_coord,
            y=y_coord,
            item=self.name,
            name=self.name,
            quantity=self.quantity,
            aux_items=self.sel_items.aux,
        )
