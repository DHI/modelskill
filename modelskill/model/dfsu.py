from __future__ import annotations
from pathlib import Path
from typing import Optional, get_args

import mikeio
import numpy as np
import pandas as pd

from ._base import SpatialField, _validate_overlap_in_time, SelectedItems
from ..types import UnstructuredType
from ..quantity import Quantity
from ..utils import _get_idx
from .point import PointModelResult
from .track import TrackModelResult
from ..obs import Observation, PointObservation, TrackObservation


class DfsuModelResult(SpatialField):
    """Construct a DfsuModelResult from a dfsu file or mikeio.Dataset/DataArray.

    Parameters
    ----------
    data : types.UnstructuredType
        the input data or file path
    name : Optional[str], optional
        The name of the model result,
        by default None (will be set to file name or item name)
    item : str | int | None, optional
        If multiple items/arrays are present in the input an item
        must be given (as either an index or a string), by default None
    quantity : Quantity, optional
        Model quantity, for MIKE files this is inferred from the EUM information
    aux_items : Optional[list[int | str]], optional
        Auxiliary items, by default None
    """

    def __init__(
        self,
        data: UnstructuredType,
        *,
        name: Optional[str] = None,
        item: str | int | None = None,
        quantity: Optional[Quantity] = None,
        aux_items: Optional[list[int | str]] = None,
    ) -> None:
        assert isinstance(
            data, get_args(UnstructuredType)
        ), "Could not construct DfsuModelResult from provided data"

        filename = None
        if isinstance(data, (str, Path)):
            assert Path(data).suffix == ".dfsu", "File must be a dfsu file"
            name = name or Path(data).stem
            filename = str(data)
            data = mikeio.open(data)

        elif isinstance(data, (mikeio.DataArray, mikeio.Dataset)):
            pass
            # we could check that geometry has FM in the name, but ideally we would like to support more or all geometries
            # if not "FM" in str(type(data.geometry)):
            #    raise ValueError(f"Geometry of {type(data.geometry)} is not supported.")
        else:
            raise ValueError(
                f"data type must be .dfsu or dfsu-Dataset/DataArray. Not {type(data)}."
            )

        if isinstance(data, mikeio.DataArray):
            if item is not None:
                raise ValueError("item must be None when data is a DataArray")
            if aux_items is not None:
                raise ValueError("aux_items must be None when data is a DataArray")
            item_info = data.item
            item = data.name
        else:
            item_names = [i.name for i in data.items]
            idx = _get_idx(x=item, valid_names=item_names)
            item_info = data.items[idx]

            sel_items = SelectedItems.parse(item_names, item=item, aux_items=aux_items)
            item = sel_items.values
            self.sel_items = sel_items

        self.data: mikeio.dfsu.Dfsu2DH | mikeio.DataArray | mikeio.Dataset = data
        self.name = name or str(item)
        self.quantity = (
            Quantity.from_mikeio_iteminfo(item_info) if quantity is None else quantity
        )
        self.filename = filename  # TODO: remove? backward compatibility

    def __repr__(self) -> str:
        # TODO add item name
        return f"<{self.__class__.__name__}> '{self.name}'"

    @property
    def time(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.data.time)

    def _in_domain(self, x: float, y: float) -> bool:
        return self.data.geometry.contains([x, y])  # type: ignore

    def extract(
        self, observation: Observation, spatial_method: Optional[str] = None
    ) -> PointModelResult | TrackModelResult:
        """Extract ModelResult at observation positions

        Note: this method is typically not called directly, but by the match() method.

        Parameters
        ----------
        observation : <PointObservation> or <TrackObservation>
            positions (and times) at which modelresult should be extracted
        spatial_method : Optional[str], optional
            spatial selection/interpolation method, 'contained' (=isel),
            'nearest', 'inverse_distance' (with 5 nearest points),
            by default None = 'inverse_distance'

        Returns
        -------
        PointModelResult or TrackModelResult
            extracted modelresult with the same geometry as the observation
        """
        method = self._parse_spatial_method(spatial_method)

        _validate_overlap_in_time(self.time, observation)
        if isinstance(observation, PointObservation):
            return self._extract_point(observation, spatial_method=method)
        elif isinstance(observation, TrackObservation):
            return self._extract_track(observation, spatial_method=method)
        else:
            raise NotImplementedError(
                f"Extraction from {type(self.data)} to {type(observation)} is not implemented."
            )

    @staticmethod
    def _parse_spatial_method(method):
        if method == "isel":
            method = "contained"
        if method == "IDW":
            method = "inverse_distance"
        if method is None:
            return None  # decide default later
        elif method not in ["nearest", "contained", "inverse_distance"]:
            raise ValueError(
                f"spatial_method for Dfsu must be 'nearest', 'contained', or 'inverse_distance'. Not {method}."
            )
        return method

    def _extract_point(
        self, observation: PointObservation, spatial_method: Optional[str] = None
    ) -> PointModelResult:
        """Spatially extract a PointModelResult from a DfsuModelResult
        given a PointObservation. No time interpolation is done!

        Note: 'inverse_distance' method uses 5 nearest points and is the default.
        """

        method = spatial_method or "inverse_distance"
        assert method in ["nearest", "contained", "inverse_distance"]
        n_nearest = 5 if method == "inverse_distance" else 1

        assert isinstance(
            self.data, (mikeio.dfsu.Dfsu2DH, mikeio.DataArray, mikeio.Dataset)
        )

        x, y = observation.x, observation.y
        if not self._in_domain(x, y):
            raise ValueError(
                f"PointObservation '{observation.name}' ({x}, {y}) outside model domain!"
            )

        if method == "contained":
            xy = np.atleast_2d([x, y])
            elemids = self.data.geometry.find_index(coords=xy)
            if isinstance(self.data, mikeio.dfsu.Dfsu2DH):
                ds_model = self.data.read(elements=elemids, items=self.sel_items.all)
            elif isinstance(self.data, mikeio.Dataset):
                ds_model = self.data[self.sel_items.all].isel(element=elemids)
            elif isinstance(self.data, mikeio.DataArray):
                da = self.data.isel(element=elemids)
                ds_model = mikeio.Dataset({da.name: da})
        else:
            if isinstance(self.data, mikeio.dfsu.Dfsu2DH):
                elemids = self.data.geometry.find_nearest_elements(
                    x, y, n_nearest=n_nearest
                )
                ds = self.data.read(elements=elemids, items=self.sel_items.all)
                ds_model = (
                    ds.interp(x=x, y=y, n_nearest=n_nearest) if n_nearest > 1 else ds
                )
            elif isinstance(self.data, mikeio.Dataset):
                ds_model = self.data[self.sel_items.all].interp(
                    x=x, y=y, n_nearest=n_nearest
                )
            elif isinstance(self.data, mikeio.DataArray):
                da = self.data.interp(x=x, y=y, n_nearest=n_nearest)
                ds_model = mikeio.Dataset({da.name: da})

        aux_items = (
            None if isinstance(self.data, mikeio.DataArray) else self.sel_items.aux
        )

        assert isinstance(ds_model, mikeio.Dataset)

        # TODO not sure why we rename here
        assert self.name is not None
        ds_model.rename({ds_model.items[0].name: self.name}, inplace=True)

        return PointModelResult(
            data=ds_model,
            item=self.name,
            x=ds_model.geometry.x,
            y=ds_model.geometry.y,
            name=self.name,
            quantity=self.quantity,
            aux_items=aux_items,
        )

    def _extract_track(
        self, observation: TrackObservation, spatial_method: Optional[str] = None
    ) -> TrackModelResult:
        """Extract a TrackModelResult from a DfsuModelResult (when data is a Dfsu object),
        given a TrackObservation.

        Wraps MIKEIO's extract_track method (which has the default method='nearest').

        MIKE IO's extract_track, inverse_distance method, uses 5 nearest points.
        """
        method = spatial_method or "inverse_distance"
        if method == "contained":
            raise NotImplementedError(
                "spatial method 'contained' (=isel) not implemented for track extraction in MIKE IO"
            )
        assert method in ["nearest", "inverse_distance"]

        assert isinstance(
            self.data, (mikeio.dfsu.Dfsu2DH, mikeio.DataArray, mikeio.Dataset)
        )

        track = observation.data.to_dataframe()

        if isinstance(self.data, mikeio.DataArray):
            ds_model = self.data.extract_track(track=track, method=method)
            ds_model.rename({self.data.name: self.name}, inplace=True)
            aux_items = None
        else:
            if isinstance(self.data, mikeio.dfsu.Dfsu2DH):
                ds_model = self.data.extract_track(
                    track=track, items=self.sel_items.all, method=method
                )
            elif isinstance(self.data, mikeio.Dataset):
                ds_model = self.data[self.sel_items.all].extract_track(
                    track=track, method=method
                )
            ds_model.rename({self.sel_items.values: self.name}, inplace=True)
            aux_items = self.sel_items.aux

        item_names = [i.name for i in ds_model.items]
        x_name = "Longitude" if "Longitude" in item_names else "x"
        y_name = "Latitude" if "Latitude" in item_names else "y"

        return TrackModelResult(
            data=ds_model.dropna(),  # TODO: not on aux cols
            item=self.name,
            x_item=x_name,
            y_item=y_name,
            name=self.name,
            quantity=self.quantity,
            aux_items=aux_items,
        )
