import os
import numpy as np
from shapely.geometry import Point
import pandas as pd
from mikeio import Dfs0


class PointObservation:

    name = None
    x = None
    y = None
    z = None
    
    df = None
    
    @property
    def time(self):
        return self.df.index

    @property
    def start_time(self):
        """First time instance (as datetime)
        """
        return self.time[0].to_pydatetime()

    @property
    def end_time(self):
        """Last time instance (as datetime)
        """
        return self.time[-1].to_pydatetime()

    @property
    def values(self):
        return self.df.values

    @property
    def n(self):
        return len(self.df)

    @property
    def geo(self) -> Point:
        if self.z is None:
            return Point(self.x, self.y)
        else:
            return Point(self.x, self.y, self.z)

    def __init__(self, filename, item: int=0, x:float=None, y:float=None, z:float=None, name=None):

        self.x = x
        self.y = y
        self.z = z
        self.name = name

        if isinstance(filename, pd.DataFrame) or isinstance(filename, pd.Series):
            raise NotImplementedError()
        else:
            if name is None:
                self.name = os.path.basename(filename).split(".")[0]

            ext = os.path.splitext(filename)[-1]
            if ext == '.dfs0':
                self.df = self._read_dfs0(Dfs0(filename), item)
            else:
                raise NotImplementedError()

    def __repr__(self):
        out = f"PointObservation: {self.name}, x={self.x}, y={self.y}"
        return out

    @staticmethod
    def from_dataframe(df):
        pass

    @staticmethod
    def from_dfs0(dfs, item_number):
        pass

    @staticmethod
    def _get_dfs_item_number(dfs, item):
        item_lookup = {item.name: i for i, item in enumerate(dfs.items)}

        if isinstance(item, str):
            i = item_lookup[item]
        elif isinstance(item, int) and item < dfs.n_items:
            i = item
        else:
            raise ValueError(f"item {item} could not be found")

        return i

    @staticmethod
    def _read_dfs0(dfs, item):
        """Read data from dfs0 file
        """
        item_number = PointObservation._get_dfs_item_number(dfs, item)
        df = dfs.read(items=item_number).to_dataframe()
        df.dropna(inplace=True)
        return df

    def subset_time(self, start=None, end=None):
        #idx = np.where(np.logical_and(self.time >= start, self.time <= end))
        self.df = self.df[start:end]

