import os
import numpy as np
from shapely.geometry import Point
import pandas as pd
from mikeio import Dfs0

class PointObservation:
    
    name = None
    filename = None
    dfs = None

    x = None
    y = None
    z = None
    item_number = None
    ds = None

    @property
    def time(self):
        return self.ds.time

    @property
    def start_time(self):
        """First time instance (as datetime)
        """
        return self.ds.time[0].to_pydatetime()

    @property
    def end_time(self):
        """Last time instance (as datetime)
        """
        return self.ds.time[-1].to_pydatetime()

    @property
    def values(self):
        return self.ds.data[0]

    @property
    def n(self):
        return len(self.time)

    @property
    def geo(self) -> Point:
        if self.z is None:
            return Point(self.x, self.y)
        else:
            return Point(self.x, self.y, self.z)        

    def __init__(self, filename, x, y, z=None, item=0, name=None):
        self.filename = filename
        self.dfs = Dfs0(filename)
        self.x = x
        self.y = y
        self.z = z

        if name is None:
            name = os.path.basename(filename).split('.')[0]
        self.name = name

        self.item_number = self._get_item_number(item)

        self.read()        

    def __repr__(self):
        out = f"PointObservation: {self.name}, x={self.x}, y={self.y}"
        return out

    def _get_item_number(self, item):
        item_lookup = {item.name: i for i, item in enumerate(self.dfs.items)}

        if isinstance(item, str):
            i = item_lookup[item]
        elif isinstance(item, int) and item < self.dfs.n_items:
            i = item
        else:
            raise ValueError(f"item {item} could not be found in {self.filename}")

        return i

    def read(self):
        """Read data from file
        """
        ds = self.dfs.read(items=self.item_number)
        ds.dropna()
        self.ds = ds

    def subset_time(self, start=None, end=None):
        idx = np.where(np.logical_and(self.time>=start, self.time<=end))
        self.ds = self.ds.isel(idx[0], axis=0)

    def to_dataframe(self):
        return self.ds.to_dataframe()