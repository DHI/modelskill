from __future__ import annotations
from pathlib import Path
from typing import Optional, get_args

import mikeio
import numpy as np
import pandas as pd

from ._base import SpatialField
from ..types import Quantity, UnstructuredType
from ..utils import _get_idx
from .point import PointModelResult
from .track import TrackModelResult
from ..observation import Observation, PointObservation, TrackObservation


class DfsuModelResult(SpatialField):
    """Construct a DfsuModelResult from a dfsu file or mikeio.Dataset/DataArray."""

    def __init__(
        self,
        data: UnstructuredType,
        *,
        name: str = "Undefined",
        item: str | int | None = None,
        quantity: Optional[Quantity] = None,
    ) -> None:
        assert isinstance(
            data, get_args(UnstructuredType)
        ), "Could not construct DfsuModelResult from provided data"

        filename = None
        if isinstance(data, (str, Path)):
            assert Path(data).suffix == ".dfsu", "File must be a dfsu file"
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
            if isinstance(item, int):
                raise ValueError("item must be a string when data is a DataArray")
            quantity = Quantity(name=repr(data.type), unit=data.unit.name)
        else:
            item_names = [i.name for i in data.items]
            idx = _get_idx(x=item, valid_names=item_names)
            item_info = data.items[idx]
            quantity = Quantity.from_mikeio_iteminfo(item_info)

        self.item = item
        self.data: mikeio.dfsu.Dfsu2DH | mikeio.DataArray | mikeio.Dataset = data
        self.name = name
        self.quantity = Quantity.undefined() if quantity is None else quantity
        self.filename = filename  # TODO: remove? backward compatibility

    def __repr__(self):
        return f"<{self.__class__.__name__}> '{self.name}'"

    @property
    def time(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.data.time)

    @property
    def start_time(self) -> pd.Timestamp:
        return self.time[0]

    @property
    def end_time(self) -> pd.Timestamp:
        return self.time[-1]

    def _in_domain(self, x, y) -> bool:
        return self.data.geometry.contains([x, y])  # type: ignore

    def extract(self, observation: Observation) -> PointModelResult | TrackModelResult:
        """Extract ModelResult at observation positions

        Parameters
        ----------
        observation : <PointObservation> or <TrackObservation>
            positions (and times) at which modelresult should be extracted

        Returns
        -------
        <modelskill.protocols.Comparable>
            A model result object with the same geometry as the observation
        """
        if isinstance(observation, PointObservation):
            return self.extract_point(observation)
        elif isinstance(observation, TrackObservation):
            return self.extract_track(observation)
        else:
            raise NotImplementedError(
                f"Extraction from {type(self.data)} to {type(observation)} is not implemented."
            )

    def extract_point(self, observation: PointObservation) -> PointModelResult:
        """Spatially extract a PointModelResult from a DfsuModelResult (when data is a Dfsu object),
        given a PointObservation. No time interpolation is done!"""

        assert isinstance(
            self.data, (mikeio.dfsu.Dfsu2DH, mikeio.DataArray, mikeio.Dataset)
        )

        x, y = observation.x, observation.y
        if not self._in_domain(x, y):
            raise ValueError(
                f"PointObservation '{observation.name}' ({x}, {y}) outside model domain!"
            )

        # TODO: interp2d
        xy = np.atleast_2d([x, y])
        elemids = self.data.geometry.find_index(coords=xy)
        if isinstance(self.data, mikeio.dfsu.Dfsu2DH):
            ds_model = self.data.read(elements=elemids, items=[self.item])
        elif isinstance(self.data, mikeio.Dataset):
            ds_model = self.data.isel(element=elemids)
        elif isinstance(self.data, mikeio.DataArray):
            da = self.data.isel(element=elemids)
            ds_model = mikeio.Dataset({da.name: da})

        # TODO not sure why we rename here
        assert self.name is not None
        ds_model.rename({ds_model.items[0].name: self.name}, inplace=True)

        return PointModelResult(
            data=ds_model,  # TODO convert to dataframe?
            x=ds_model.geometry.x,
            y=ds_model.geometry.y,
            name=self.name,
            quantity=self.quantity,
        )

    def extract_track(self, observation: TrackObservation) -> TrackModelResult:
        """Extract a TrackModelResult from a DfsuModelResult (when data is a Dfsu object),
        given a TrackObservation."""

        assert isinstance(
            self.data, (mikeio.dfsu.Dfsu2DH, mikeio.DataArray, mikeio.Dataset)
        )

        # TODO: data could be xarray
        track = (
            observation.data
            if isinstance(observation.data, pd.DataFrame)
            else observation.data.to_dataframe()
        )

        if isinstance(self.data, mikeio.dfsu.Dfsu2DH):
            ds_model = self.data.extract_track(track=track, items=[self.item])
        elif isinstance(self.data, (mikeio.Dataset, mikeio.DataArray)):
            ds_model = self.data.extract_track(track=track)
        ds_model.rename({ds_model.items[-1].name: self.name}, inplace=True)

        return TrackModelResult(
            data=ds_model.dropna(),  # .to_dataframe().dropna(),
            item=self.name,
            name=self.name,
            quantity=self.quantity,
        )
