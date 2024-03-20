from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, get_args

import pandas as pd
import xarray as xr

from ._base import SpatialField, _validate_overlap_in_time, SelectedItems
from ..utils import rename_coords_xr, rename_coords_pd
from ..types import GridType
from ..quantity import Quantity
from .point import PointModelResult
from .track import TrackModelResult
from ..obs import PointObservation, TrackObservation


class GridModelResult(SpatialField):
    """Construct a GridModelResult from a file or xarray.Dataset.

    Parameters
    ----------
    data : types.GridType
        the input data or file path
    name : str, optional
        The name of the model result,
        by default None (will be set to file name or item name)
    item : str or int, optional
        If multiple items/arrays are present in the input an item
        must be given (as either an index or a string), by default None
    quantity : Quantity, optional
        Model quantity, for MIKE files this is inferred from the EUM information
    aux_items : Optional[list[int | str]], optional
        Auxiliary items, by default None
    """

    def __init__(
        self,
        data: GridType,
        *,
        name: Optional[str] = None,
        item: str | int | None = None,
        quantity: Optional[Quantity] = None,
        aux_items: Optional[list[int | str]] = None,
    ) -> None:
        assert isinstance(
            data, get_args(GridType)
        ), "Could not construct GridModelResult from provided data."

        if isinstance(data, (str, Path)):
            if "*" in str(data):
                ds = xr.open_mfdataset(data)
            else:
                assert Path(data).exists(), f"{data}: File does not exist."
                ds = xr.open_dataset(data)

        elif isinstance(data, Sequence) and all(
            isinstance(file, (str, Path)) for file in data
        ):
            ds = xr.open_mfdataset(data)

        elif isinstance(data, xr.DataArray):
            if item is not None:
                raise ValueError(f"item must be None when data is a {type(data)}")
            if aux_items is not None:
                raise ValueError(f"aux_items must be None when data is a {type(data)}")
            if data.ndim < 2:
                raise ValueError(f"DataArray must at least 2D. Got {list(data.dims)}.")
            ds = data.to_dataset(name=name, promote_attrs=True)
        elif isinstance(data, xr.Dataset):
            assert len(data.coords) >= 2, "Dataset must have at least 2 dimensions."
            ds = data
        else:
            raise NotImplementedError(
                f"Could not construct GridModelResult from {type(data)}"
            )

        sel_items = SelectedItems.parse(
            list(ds.data_vars), item=item, aux_items=aux_items
        )
        name = name or sel_items.values
        ds = rename_coords_xr(ds)

        self.data: xr.Dataset = ds[sel_items.all]
        self.name = name
        self.sel_items = sel_items

        # use long_name and units from data if not provided
        if quantity is None:
            da = self.data[sel_items.values]
            quantity = Quantity.from_cf_attrs(da.attrs)

        self.quantity = quantity

    def __repr__(self) -> str:
        res = []
        res.append(f"<{self.__class__.__name__}>: {self.name}")
        res.append(f"Time: {self.time[0]} - {self.time[-1]}")
        res.append(f"Quantity: {self.quantity}")
        if len(self.sel_items.aux) > 0:
            res.append(f"Auxiliary variables: {', '.join(self.sel_items.aux)}")
        return "\n".join(res)

    @property
    def time(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.data.time)

    def _in_domain(self, x: float, y: float) -> bool:
        assert hasattr(self.data, "x") and hasattr(
            self.data, "y"
        ), "Data has no x and/or y coordinates."
        xmin = float(self.data.x.values.min())
        xmax = float(self.data.x.values.max())
        ymin = float(self.data.y.values.min())
        ymax = float(self.data.y.values.max())
        return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    def extract(
        self,
        observation: PointObservation | TrackObservation,
        spatial_method: Optional[str] = None,
    ) -> PointModelResult | TrackModelResult:
        """Extract ModelResult at observation positions

        Note: this method is typically not called directly, but through the match() method.

        Parameters
        ----------
        observation : <PointObservation> or <TrackObservation>
            positions (and times) at which modelresult should be extracted
        spatial_method : Optional[str], optional
            method in xarray.Dataset.interp, typically either "nearest" or
            "linear", by default None = 'linear'

        Returns
        -------
        PointModelResult or TrackModelResult
            extracted modelresult
        """
        _validate_overlap_in_time(self.time, observation)
        if isinstance(observation, PointObservation):
            return self._extract_point(observation, spatial_method)
        elif isinstance(observation, TrackObservation):
            return self._extract_track(observation, spatial_method)
        else:
            raise NotImplementedError(
                f"Extraction from {type(self.data)} to {type(observation)} is not implemented."
            )

    def _extract_point(
        self, observation: PointObservation, spatial_method: Optional[str] = None
    ) -> PointModelResult:
        """Spatially extract a PointModelResult from a GridModelResult (when data is a xarray.Dataset),
        given a PointObservation. No time interpolation is done!"""
        method: str = spatial_method or "linear"

        x, y, z = observation.x, observation.y, observation.z
        if (x is None) or (y is None):
            raise ValueError(
                f"PointObservation '{observation.name}' cannot be used for extraction "
                + f"because it has None position x={x}, y={y}. Please provide position "
                + "when creating PointObservation."
            )
        if not self._in_domain(x, y):
            raise ValueError(
                f"PointObservation '{observation.name}' ({x}, {y}) is outside model domain!"
            )

        assert isinstance(self.data, xr.Dataset)

        # TODO: avoid runtrip to pandas if possible (potential loss of metadata)
        if "z" in self.data.dims and z is not None:
            ds = self.data.interp(
                coords=dict(x=x, y=y, z=z), method=method  # type: ignore
            )
        else:
            ds = self.data.interp(coords=dict(x=x, y=y), method=method)  # type: ignore
        # TODO: exclude aux cols in dropna
        df = ds.to_dataframe().drop(columns=["x", "y"]).dropna()
        if len(df) == 0:
            raise ValueError(
                f"Spatial point extraction failed for PointObservation '{observation.name}' in GridModelResult '{self.name}'! (is point outside model domain? Consider spatial_method='nearest')"
            )
        df = df.rename(columns={self.sel_items.values: self.name})

        return PointModelResult(
            data=df,
            x=ds.x.item(),
            y=ds.y.item(),
            item=self.name,
            name=self.name,
            quantity=self.quantity,
            aux_items=self.sel_items.aux,
        )

    def _extract_track(
        self, observation: TrackObservation, spatial_method: Optional[str] = None
    ) -> TrackModelResult:
        """Extract a TrackModelResult from a GridModelResult (when data is a xarray.Dataset),
        given a TrackObservation."""
        method: str = spatial_method or "linear"

        obs_df = observation.data.to_dataframe()

        renamed_obs_data = rename_coords_pd(obs_df)
        t = xr.DataArray(renamed_obs_data.index, dims="track")
        x = xr.DataArray(renamed_obs_data.x, dims="track")
        y = xr.DataArray(renamed_obs_data.y, dims="track")

        assert isinstance(self.data, xr.Dataset)
        ds = self.data.interp(
            coords=dict(time=t, x=x, y=y),
            method=method,  # type: ignore
        )
        df = ds.to_dataframe().drop(columns=["time"])
        df = df.rename(columns={self.sel_items.values: self.name})

        return TrackModelResult(
            data=df.dropna(),  # TODO: exclude aux cols in dropna
            item=self.name,
            x_item="x",
            y_item="y",
            name=self.name,
            quantity=self.quantity,
            aux_items=self.sel_items.aux,
        )
