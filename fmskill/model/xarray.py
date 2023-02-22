import os
import pandas as pd
import warnings

from ..observation import Observation, PointObservation, TrackObservation
from ..comparison import PointComparer, TrackComparer
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

    @staticmethod
    def _validate_coord_names(coords):
        cnames = [c for c in coords]
        for c in ["x", "y", "time"]:
            if c not in coords:
                raise ValueError(f"{c} not found in coords {cnames}")

    def _validate_time_axis(self, coords):
        if "time" not in coords:
            raise ValueError(
                f"Time coordinate could not be found in {[c for c in coords]}"
            )
        if not isinstance(coords["time"].to_index(), pd.DatetimeIndex):
            raise ValueError(f"Time coordinate is not equivalent to DatetimeIndex")
        # return coords["time"].dtype.type == np.datetime64

    def _get_item_name(self, item, item_names=None) -> str:
        if item_names is None:
            item_names = list(self.ds.data_vars)
        n_items = len(item_names)
        if item is None:
            if n_items == 1:
                return item_names[0]
            else:
                return None
        # if isinstance(item, mikeio.ItemInfo):
        #     item = item.name
        if isinstance(item, int):
            if item < 0:  # Handle negative indices
                item = n_items + item
            if (item < 0) or (item >= n_items):
                raise IndexError(f"item {item} out of range (0, {n_items-1})")
            item = item_names[item]
        elif isinstance(item, str):
            if item not in item_names:
                raise KeyError(f"item must be one of {item_names}")
        else:
            raise TypeError("item must be int or string")
        return item

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

    def __init__(self, data, name: str = None, item=None, itemInfo=None, **kwargs):
        import xarray as xr

        self.itemInfo = _parse_itemInfo(itemInfo)

        self._filename = None
        if isinstance(data, str) and ("*" not in data):
            self._filename = data
            ds = xr.open_dataset(data, **kwargs)
            if name is None:
                name = os.path.basename(data).split(".")[0]
        elif isinstance(data, str) or isinstance(data, list):
            # multi-file dataset; input is list of filenames or str with wildcard
            self._filename = data if isinstance(data, str) else ";".join(data)
            ds = xr.open_mfdataset(data, **kwargs)
        elif isinstance(data, xr.Dataset):
            ds = data
            # TODO: name
        elif isinstance(data, xr.DataArray):
            ds = data.to_dataset()
            # TODO: name
        else:
            raise TypeError(
                f"Unknown input type {type(data)}. Must be str or xarray.Dataset/DataArray."
            )

        if item is None:
            if len(ds.data_vars) == 1:
                item = list(ds.data_vars)[0]
            else:
                raise ValueError(f"Model ambiguous - please provide item! Available items: {list(ds.data_vars)}")

        ds = self._rename_coords(ds)
        self._validate_coord_names(ds.coords)
        self._validate_time_axis(ds.coords)

        item = self._get_item_name(item, list(ds.data_vars))
        self.ds = ds[[item]]
        self._selected_item = item
        if name is None:
            name = self.item_name

        self.name = name

    def _rename_coords(self, ds):
        new_names = self._get_new_coord_names(ds.coords)
        if len(new_names) > 0:
            ds = ds.rename(new_names)
        return ds

    def extract_observation(
        self, observation: PointObservation, **kwargs
    ) -> PointComparer:
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
            comparer = PointComparer(observation, df_model, **kwargs)
        elif isinstance(observation, TrackObservation):
            df_model = self._extract_track(observation, item)
            comparer = TrackComparer(observation, df_model, **kwargs)
        else:
            raise ValueError("Only point and track observation are supported!")

        if len(comparer.df) == 0:
            warnings.warn(f"No overlapping data in found for obs '{observation.name}'!")
            comparer = None

        return comparer
