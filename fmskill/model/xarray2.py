import os
from typing import Union
import pandas as pd
from dataclasses import dataclass

from ..observation import PointObservation, TrackObservation
from ..comparison import BaseComparer, PointComparer, TrackComparer
from .abstract import ModelResultInterface


@dataclass
class XArrayModelResult(ModelResultInterface):
    import xarray as xr

    name: str
    ds: xr.Dataset

    @staticmethod
    def create_from_file(filename: str, *, name=None, **kwargs) -> "XArrayModelResult":
        import xarray as xr

        ds = xr.open_dataset(filename, **kwargs)
        if name is None:
            name = os.path.basename(filename).split(".")[0]
        return XArrayModelResult(name=name, ds=ds)

    def __len__(self):
        # Is this useful?
        return len(self.ds)

    @property
    def item_names(self):
        # Is this useful?
        """List of item names (=data vars)"""
        return list(self.ds.data_vars)

    def validate(self) -> None:
        self._validate_coord_names(self.ds.coords)
        self._validate_time_axis(self.ds.coords)

    def __post_init__(self):
        self._rename_coords()
        self.validate()

    def _rename_coords(self) -> None:
        new_names = self._get_new_coord_names(self.ds.coords)
        self.ds = self.ds.rename(new_names)

    def item_name(self):
        return self.item

    @staticmethod
    def _get_new_coord_names(coords):
        new_names = {}
        for coord in coords:
            c = coord.lower()
            if ("x" not in new_names) and (("lon" in c) or ("east" in c)):
                new_names[coord] = "x"
            elif ("y" not in new_names) and (("lat" in c) or ("north" in c)):
                new_names[coord] = "y"
            elif ("time" not in new_names) and ("date" in c):
                new_names[coord] = "time"
        return new_names

    @staticmethod
    def _validate_coord_names(coords):
        cnames = list(coords)
        for c in ["x", "y", "time"]:
            if c not in coords:
                raise ValueError(f"{c} not found in coords {cnames}")

    @staticmethod
    def _validate_time_axis(coords):
        if "time" not in coords:
            raise ValueError(
                f"Time coordinate could not be found in {[c for c in coords]}"
            )
        if not isinstance(coords["time"].to_index(), pd.DatetimeIndex):
            raise ValueError(f"Time coordinate is not equivalent to DatetimeIndex")

    @property
    def start_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.ds.time.values[0])

    @property
    def end_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.ds.time.values[-1])

    def extract_observation(
        self, observation: Union[PointObservation, TrackObservation], item: str
    ) -> BaseComparer:
        """Compare this ModelResult with an observation

        Parameters
        ----------
        observation : <PointObservation>
            Observation to be compared

        Returns
        -------
        <fmskill.PointComparer>
            A comparer object for further analysis or plotting
        """

        if isinstance(observation, PointObservation):
            df_model = self._extract_point(observation, item)
            return PointComparer(observation, df_model)
        elif isinstance(observation, TrackObservation):
            df_model = self._extract_track(observation, item)
            return TrackComparer(observation, df_model)
        else:
            raise ValueError("Only point and track observation are supported!")

    def _extract_point(self, observation: PointObservation, item: str) -> pd.DataFrame:
        x, y = observation.x, observation.y
        if (x is None) or (y is None):
            raise ValueError(
                f"PointObservation '{observation.name}' cannot be used for extraction "
                + f"because it has None position x={x}, y={y}. Please provide position "
                + "when creating PointObservation."
            )
        da = self.ds[item].interp(coords=dict(x=x, y=y), method="nearest")
        df = da.to_dataframe().drop(columns=["x", "y"])
        df = df.rename(columns={df.columns[-1]: self.name})
        return df.dropna()

    def _extract_track(self, observation: TrackObservation, item: str) -> pd.DataFrame:
        import xarray as xr

        t = xr.DataArray(observation.df.index, dims="track")
        x = xr.DataArray(observation.df.Longitude, dims="track")
        y = xr.DataArray(observation.df.Latitude, dims="track")
        da = self.ds[item].interp(coords=dict(time=t, x=x, y=y), method="linear")
        df = da.to_dataframe().drop(columns=["time"])
        df.index.name = "time"
        df = df.rename(columns={df.columns[-1]: self.name})
        return df.dropna()
