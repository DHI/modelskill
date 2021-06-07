import os
import numpy as np
import pandas as pd
import xarray as xr
import warnings

from mikeio import eum
from ..observation import PointObservation
from ..comparison import PointComparer
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

    @staticmethod
    def _get_new_coord_names(coords):
        new_names = {}
        for c in coords:
            clow = c.lower()
            if ("lon" in clow) or ("east" in clow) or ("x" in clow):
                new_names[c] = "x"
            if ("lat" in clow) or ("north" in clow) or ("y" in clow):
                new_names[c] = "y"
            if ("time" in clow) or ("date" in clow):
                new_names[c] = "time"
        return new_names

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
            if (item < 0) or (item >= n_items):
                raise ValueError(f"item must be between 0 and {n_items-1}")
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


class XArrayModelResultItem(_XarrayBase, ModelResultInterface):
    @property
    def item_name(self):
        return self._selected_item

    def __init__(self, ds, name: str = None, item=None):
        if isinstance(ds, (xr.DataArray, xr.Dataset)):
            self._validate_time_axis(ds)
        else:
            raise TypeError("Input must be xarray Dataset or DataArray!")

        if item is None:
            if len(ds.data_vars) == 1:
                item = list(ds.data_vars)[0]
            else:
                raise ValueError("Model ambiguous - please provide item")

        self.ds = ds
        self._selected_item = self._get_item_name(item)
        self.name = name

    def __repr__(self):
        txt = [f"<XarrayModelResultItem> '{self.name}'"]
        txt.append(f"- Item: {self.item_name}")
        return "\n".join(txt)

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
        raise NotImplementedError()
        if isinstance(observation, PointObservation):
            pass
        else:
            raise ValueError("Only point observation are supported!")

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
        if isinstance(input, str):
            self._filename = input
            ds = xr.open_dataset(input, **kwargs)
            if name is None:
                name = os.path.basename(input).split(".")[0]
        elif isinstance(input, xr.Dataset):
            ds = input
            self._filename = None
        elif isinstance(input, xr.DataArray):
            ds = input.to_dataset()
            self._filename = None
        else:
            raise TypeError(
                f"Unknown input type {type(input)}. Must be str or xarray.Dataset/DataArray."
            )

        ds = self._rename_coords(ds)
        self._validate_time_axis(ds.coords)
        self.ds = ds
        self.name = name

        self._mr_items = {}
        for it in self.item_names:
            # TODO: add filename to args
            self._mr_items[it] = XArrayModelResultItem(self.ds, self.name, it)

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

    def __repr__(self):
        txt = [f"<XarrayModelResult> '{self.name}'"]
        for j, item in enumerate(self.item_names):
            txt.append(f"- Item: {j}: {item}")
        return "\n".join(txt)
