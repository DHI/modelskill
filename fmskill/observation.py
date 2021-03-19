import os
from shapely.geometry import Point, MultiPoint
import pandas as pd
from mikeio import Dfs0, eum


class Observation:
    # name = None
    # df = None
    # itemInfo = None
    color = "#d62728"

    # DHI: darkblue: #004165,
    #      midblue:  #0098DB,
    #      gray:     #8B8D8E,
    #      lightblue:#63CEFF,
    # DHI secondary
    #      yellow:   #FADC41,
    #      orange:   #FF8849
    #      lightblue2:#C1E2E5
    #      green:    #61C250
    #      purple:   #93509E
    #      darkgray: #51626F

    # matplotlib: red=#d62728

    @property
    def time(self):
        return self.df.index

    @property
    def start_time(self):
        """First time instance (as datetime)"""
        return self.time[0].to_pydatetime()

    @property
    def end_time(self):
        """Last time instance (as datetime)"""
        return self.time[-1].to_pydatetime()

    @property
    def values(self):
        return self.df.values

    @property
    def n_points(self):
        """Number of observations"""
        return len(self.df)

    def __init__(self, name: str = None):
        self.name = name

    def _unit_text(self):
        if self.itemInfo is None:
            return ""
        txt = f"{self.itemInfo.type.display_name}"
        if self.itemInfo.type != eum.EUMType.Undefined:
            unit = self.itemInfo.unit.display_name
            txt = f"{txt} [{unit_display_name(unit)}]"
        return txt

    def hist(self, bins=100, **kwargs):
        """plot histogram"""
        ax = self.df.iloc[:, -1].hist(bins=bins, color=self.color, **kwargs)
        ax.set_title(self.name)
        ax.set_xlabel(self._unit_text())
        return ax


class PointObservation(Observation):

    x = None
    y = None
    z = None

    @property
    def geometry(self) -> Point:
        """Coordinates of observation"""
        if self.z is None:
            return Point(self.x, self.y)
        else:
            return Point(self.x, self.y, self.z)

    def __init__(
        self,
        filename,
        item: int = 0,
        x: float = None,
        y: float = None,
        z: float = None,
        name=None,
    ):
        self.x = x
        self.y = y
        self.z = z

        if isinstance(filename, pd.DataFrame) or isinstance(filename, pd.Series):
            raise NotImplementedError()
        else:
            if name is None:
                name = os.path.basename(filename).split(".")[0]

            ext = os.path.splitext(filename)[-1]
            if ext == ".dfs0":
                df, itemInfo = self._read_dfs0(Dfs0(filename), item)
                self.df, self.itemInfo = df, itemInfo
            else:
                raise NotImplementedError()

        super().__init__(name)

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
    def _read_dfs0(dfs, item):
        """Read data from dfs0 file"""
        df = dfs.read(items=item).to_dataframe()
        df.dropna(inplace=True)
        return df, dfs.items[item]

    def plot(self, **kwargs):
        """plot timeseries"""
        ax = self.df.plot(marker=".", color=self.color, linestyle="None", **kwargs)
        ax.set_title(self.name)
        ax.set_ylabel(self._unit_text())
        return ax


class TrackObservation(Observation):
    @property
    def geometry(self) -> MultiPoint:
        """Coordinates of observation"""
        return MultiPoint(self.df.iloc[:, 0:2].values)

    @property
    def x(self):
        return self.df.iloc[:, 0].values

    @property
    def y(self):
        return self.df.iloc[:, 1].values

    @property
    def values(self):
        return self.df.iloc[:, 2].values

    def __init__(self, filename, item: int = 2, name=None):
        if isinstance(filename, pd.DataFrame):  # or isinstance(filename, pd.Series):
            df = filename
            self.df = df.iloc[:, [0, 1, item]]
            self.itemInfo = eum.ItemInfo(eum.EUMType.Undefined)
        else:
            if name is None:
                name = os.path.basename(filename).split(".")[0]

            ext = os.path.splitext(filename)[-1]
            if ext == ".dfs0":
                items = [0, 1, item]
                df, itemInfo = self._read_dfs0(Dfs0(filename), items)
                self.df, self.itemInfo = df, itemInfo
            else:
                raise NotImplementedError()

        super().__init__(name)

    def __repr__(self):
        out = f"TrackObservation: {self.name}, n={self.n_points}"
        return out

    @staticmethod
    def _read_dfs0(dfs, items):
        """Read track data from dfs0 file"""
        df = dfs.read(items=items).to_dataframe()
        df.dropna(inplace=True)
        return df, dfs.items[items[-1]]


def unit_display_name(name: str) -> str:
    """Display name

    Examples
    --------
    >>> unit_display_name("meter")
    m
    """

    res = name.replace("meter", "m").replace("_per_", "/").replace("sec", "s")

    return res
