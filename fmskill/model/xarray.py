import warnings

import pandas as pd
import xarray as xr

from ..comparison import PointComparer, TrackComparer
from ..observation import Observation, PointObservation, TrackObservation
from .abstract import ModelResultInterface, _parse_itemInfo


class _XarrayBase:
    @property
    def start_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.ds.time.values[0])

    @property
    def end_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.ds.time.values[-1])

    @property
    def filename(self):
        return self._filename

    @staticmethod
    def _get_new_coord_names(coords):
        new_names = {}
        for coord in coords:
            c = coord.lower()
            if ("x" not in new_names) and (("lon" in c) or ("east" in c)):
                new_names[coord] = "x"
            elif ("y" not in new_names) and (("lat" in c) or ("north" in c)):
                new_names[coord] = "y"
            elif ("time" not in new_names) and (("time" in c) or ("date" in c)):
                new_names[coord] = "time"
        return new_names

    def _get_item_num(self, item) -> int:
        item_name = self._get_item_name(item)
        item_names = list(self.ds.data_vars)
        return item_names.index(item_name)

    def _extract_point(self, observation: PointObservation, item=None) -> pd.DataFrame:
        if item is None:
            item = self._selected_item
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

    def _extract_track(self, observation: TrackObservation, item=None) -> pd.DataFrame:
        import xarray as xr

        if item is None:
            item = self._selected_item
        t = xr.DataArray(observation.df.index, dims="track")
        x = xr.DataArray(observation.df.Longitude, dims="track")
        y = xr.DataArray(observation.df.Latitude, dims="track")
        da = self.ds[item].interp(coords=dict(time=t, x=x, y=y), method="linear")
        df = da.to_dataframe().drop(columns=["time"])
        df.index.name = "time"
        df = df.rename(columns={df.columns[-1]: self.name})
        return df.dropna()

    def _in_domain(self, x, y) -> bool:
        if (x is None) or (y is None):
            raise ValueError(
                "PointObservation has None position - cannot determine if inside xarray domain!"
            )
        xmin = self.ds.x.values.min()
        xmax = self.ds.x.values.max()
        ymin = self.ds.y.values.min()
        ymax = self.ds.y.values.max()
        return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    def _validate_start_end(self, observation: Observation) -> bool:
        if observation.end_time < self.start_time:
            return False
        if observation.start_time > self.end_time:
            return False
        return True


class XArrayModelResultItem(_XarrayBase, ModelResultInterface):
    @property
    def item_name(self):
        return self._selected_item

    def __init__(
        self,
        da: xr.DataArray,
        name: str = None,
        itemInfo=None,
        filename=None,
    ):
        self.itemInfo = _parse_itemInfo(itemInfo)

        self.da = da
        self._selected_item = self.da.name
        if name is None:
            name = self.item_name
        self.name = name
        self._filename = filename

    def extract_observation(self, observation: PointObservation) -> PointComparer:
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
        item = self._selected_item

        # TODO: more validation?

        if isinstance(observation, PointObservation):
            df_model = self._extract_point(observation, item)
            comparer = PointComparer(observation, df_model)
        elif isinstance(observation, TrackObservation):
            df_model = self._extract_track(observation, item)
            comparer = TrackComparer(observation, df_model)
        else:
            raise ValueError("Only point and track observation are supported!")

        if len(comparer.df) == 0:
            warnings.warn(f"No overlapping data in found for obs '{observation.name}'!")
            comparer = None

        return comparer
