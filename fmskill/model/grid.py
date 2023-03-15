from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union, get_args
import warnings

import pandas as pd
import xarray as xr

from fmskill import types, utils
from fmskill.model import protocols
from fmskill.model._base import ModelResultBase
from fmskill.model.point import PointModelResult
from fmskill.model.track import TrackModelResult
from fmskill.observation import Observation, PointObservation, TrackObservation


class GridModelResult(ModelResultBase):
    def __init__(
        self,
        data: types.GridType,
        *,
        name: Optional[str] = None,
        item: Optional[Union[str, int]] = None,
        itemInfo=None,
        quantity: Optional[str] = None,
    ) -> None:
        assert isinstance(
            data, get_args(types.GridType)
        ), "Could not construct GridModelResult from provided data."

        name = name or super()._default_name(data)

        if isinstance(data, (str, Path)):
            if "*" in str(data):
                data = xr.open_mfdataset(data)
            else:
                assert Path(data).exists(), f"{data}: File does not exist."
                data = xr.open_dataset(data)

        elif isinstance(data, Sequence) and all(
            isinstance(file, (str, Path)) for file in data
        ):
            data = xr.open_mfdataset(data)

        elif isinstance(data, xr.DataArray):
            assert data.ndim >= 2, "DataArray must at least 2D."
            data = data.to_dataset(name=name, promote_attrs=True)
        elif isinstance(data, xr.Dataset):
            assert len(data.coords) >= 2, "Dataset must have at least 2 dimensions."

        else:
            raise NotImplementedError(
                f"Could not construct GridModelResult from {type(data)}"
            )

        item, _ = utils.get_item_name_and_idx(list(data.data_vars), item)
        name = name or item
        data = utils.rename_coords_xr(data)

        super().__init__(
            data=data, name=name, item=item, itemInfo=itemInfo, quantity=quantity
        )

    def _in_domain(self, x, y) -> bool:
        if (x is None) or (y is None):
            raise ValueError(f"Cannot determine if point ({x}, {y}) is inside domain!")
        assert hasattr(self.data, "x") and hasattr(
            self.data, "y"
        ), "Data has no x and/or y coordinates."
        xmin = self.data.x.values.min()
        xmax = self.data.x.values.max()
        ymin = self.data.y.values.min()
        ymax = self.data.y.values.max()
        return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    def extract(self, observation: Observation) -> protocols.Comparable:
        extractor_lookup: Mapping[Observation, Callable] = {
            PointObservation: self._extract_point,
            TrackObservation: self._extract_track,
        }
        extraction_func = extractor_lookup.get(type(observation))
        if extraction_func is None:
            raise NotImplementedError(
                f"Extraction from {type(self.data)} to {type(observation)} is not implemented."
            )
        return extraction_func(observation)

    @staticmethod
    def _any_obs_in_model_time(
        time_obs: pd.DatetimeIndex, time_model: pd.DatetimeIndex
    ) -> bool:
        """Check if any observation times are in model time range"""
        return (time_obs[-1] >= time_model[0]) & (time_obs[0] <= time_model[-1])

    def _validate_any_obs_in_model_time(
        self, obs_name: str, time_obs: pd.DatetimeIndex, time_model: pd.DatetimeIndex
    ) -> None:
        """Check if any observation times are in model time range"""
        ok = self._any_obs_in_model_time(time_obs, time_model)
        if not ok:
            # raise ValueError(
            warnings.warn(
                f"No time overlap. Observation '{obs_name}' outside model time range! "
                + f"({time_obs[0]} - {time_obs[-1]}) not in ({time_model[0]} - {time_model[-1]})"
            )

    def _extract_point(self, observation: PointObservation) -> PointModelResult:
        """Spatially extract a PointModelResult from a GridModelResult (when data is a xarray.Dataset),
        given a PointObservation. No time interpolation is done!"""

        x, y = observation.x, observation.y
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

        self._validate_any_obs_in_model_time(
            observation.name, observation.data.index, self.time
        )

        da = self.data[self.item].interp(coords=dict(x=x, y=y), method="nearest")
        df = da.to_dataframe().drop(columns=["x", "y"])
        df = df.rename(columns={df.columns[-1]: self.name})

        return PointModelResult(
            data=df.dropna(),
            x=da.x.item(),
            y=da.y.item(),
            item=self.name,
            itemInfo=self.itemInfo,
            name=self.name,
            quantity=self.quantity,
        )

    def _extract_track(self, observation: TrackObservation) -> TrackModelResult:
        """Extract a TrackModelResult from a GridModelResult (when data is a xarray.Dataset),
        given a TrackObservation."""

        self._validate_any_obs_in_model_time(
            observation.name, observation.data.index, self.time
        )

        renamed_obs_data = utils.rename_coords_pd(observation.data)
        t = xr.DataArray(renamed_obs_data.index, dims="track")
        x = xr.DataArray(renamed_obs_data.x, dims="track")
        y = xr.DataArray(renamed_obs_data.y, dims="track")
        da = self.data[self.item].interp(coords=dict(time=t, x=x, y=y), method="linear")
        df = da.to_dataframe().drop(columns=["time"])
        # df.index.name = "time"
        df = df.rename(columns={df.columns[-1]: self.name})

        return TrackModelResult(
            data=df.dropna(),
            item=self.name,
            itemInfo=self.itemInfo,
            name=self.name,
            quantity=self.quantity,
        )
