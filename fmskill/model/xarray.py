import os
import numpy as np
import pandas as pd
import warnings

from mikeio import eum
from ..observation import Observation, PointObservation, TrackObservation
from ..comparison import PointComparer, TrackComparer
from .abstract import ModelResultInterface, MultiItemModelResult


class _XarrayBase:
    ds = None
    itemInfo = eum.ItemInfo(eum.EUMType.Undefined)

    @property
    def start_time(self):
        return pd.Timestamp(self.ds.time.values[0])

    @property
    def end_time(self):
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
        # if isinstance(item, eum.ItemInfo):
        #     item = item.name
        if isinstance(item, int):
            if item < 0:
                item = n_items + item
            if (item < 0) or (item >= n_items):
                raise ValueError(
                    f"item must be between 0 and {n_items-1} (or {-n_items} and -1)"
                )
            item = item_names[item]
        elif isinstance(item, str):
            if item not in item_names:
                raise ValueError(f"item must be one of {item_names}")
        else:
            raise ValueError("item must be int or string")
        return item

    def _get_item_num(self, item) -> int:
        item_name = self._get_item_name(item)
        item_names = list(self.ds.data_vars)
        return item_names.index(item_name)

    def _extract_point(self, observation: PointObservation, item=None) -> pd.DataFrame:
        if item is None:
            item = self._selected_item
        x, y = observation.x, observation.y
        da = self.ds[item].interp(coords=dict(x=x, y=y), method="nearest")
        df = da.to_dataframe().drop(columns=["x", "y"])
        df = df.rename(columns={df.columns[-1]: self.name})
        return df

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
        return df

    def _in_domain(self, x, y) -> bool:
        ok = True
        # TODO
        # if self.is_dfsu:
        #    ok = self.dfs.contains([x, y])
        return ok

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

    def __init__(self, ds, name: str = None, item=None, filename=None):
        import xarray as xr

        if isinstance(ds, (xr.DataArray, xr.Dataset)):
            self._validate_time_axis(ds)
        else:
            raise TypeError("Input must be xarray Dataset or DataArray!")

        if item is None:
            if len(ds.data_vars) == 1:
                item = list(ds.data_vars)[0]
            else:
                raise ValueError("Model ambiguous - please provide item")

        item = self._get_item_name(item, list(ds.data_vars))
        self.ds = ds[[item]]
        self._selected_item = item
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


class XArrayModelResult(_XarrayBase, MultiItemModelResult):
    @property
    def item_names(self):
        """List of item names (=data vars)"""
        return list(self.ds.data_vars)

    def __init__(self, input, name: str = None, item=None, **kwargs):
        import xarray as xr

        self._filename = None
        if isinstance(input, str) and ("*" not in input):
            self._filename = input
            ds = xr.open_dataset(input, **kwargs)
            if name is None:
                name = os.path.basename(input).split(".")[0]
        elif isinstance(input, str) or isinstance(input, list):
            # multi-file dataset; input is list of filenames or str with wildcard
            self._filename = input if isinstance(input, str) else ";".join(input)
            ds = xr.open_mfdataset(input, **kwargs)
        elif isinstance(input, xr.Dataset):
            ds = input
            # TODO: name
        elif isinstance(input, xr.DataArray):
            ds = input.to_dataset()
            # TODO: name
        else:
            raise TypeError(
                f"Unknown input type {type(input)}. Must be str or xarray.Dataset/DataArray."
            )

        ds = self._rename_coords(ds)
        self._validate_coord_names(ds.coords)
        self._validate_time_axis(ds.coords)
        self.ds = ds
        self.name = name

        self._mr_items = {}
        for it in self.item_names:
            self._mr_items[it] = XArrayModelResultItem(
                self.ds, self.name, it, self._filename
            )

        if item is not None:
            self._selected_item = self._get_item_name(item)
        elif len(self.item_names) == 1:
            self._selected_item = 0
        else:
            self._selected_item = None

    def _rename_coords(self, ds):
        new_names = self._get_new_coord_names(ds.coords)
        if len(new_names) > 0:
            ds = ds.rename(new_names)
        return ds
