"""
The `observation` module contains different types of Observation classes for
fixed locations (PointObservation), or locations moving in space (TrackObservation).

Examples
--------
>>> o1 = PointObservation("klagshamn.dfs0", item=0, x=366844, y=6154291, name="Klagshamn")
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
import mikeio
from copy import deepcopy
from .utils import make_unique_index


def _parse_item(items, item, item_str="item"):
    if isinstance(item, int):
        item = len(items) + item if (item < 0) else item
        if (item < 0) or (item >= len(items)):
            raise IndexError(f"{item_str} is out of range (0, {len(items)})")
    elif isinstance(item, str):
        item = items.index(item)
    else:
        raise TypeError(f"{item_str} must be int or string")
    return item


class Observation:
    "Base class for all types of observations"

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

    def __init__(
        self, name: str = None, df=None, itemInfo=None, variable_name: str = None, override_units: str = None,
    ):
        self.color = "#d62728"

        if name is None:
            name = "Observation"
        self.name = name
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                f"Input must have a datetime index! Provided index was {type(df.index)}"
            )
        time = df.index.round(freq="100us")  # 0.0001s accuracy
        df.index = pd.DatetimeIndex(time, freq="infer")
        self.df = df
        if itemInfo is None:
            self.itemInfo = mikeio.ItemInfo(mikeio.EUMType.Undefined)
        else:
            self.itemInfo = itemInfo
        self.weight = 1.0
        if variable_name is None:
            variable_name = self.itemInfo.type.name
        self.variable_name = variable_name
        self.override_units= override_units
    @property
    def time(self) -> pd.DatetimeIndex:
        "Time index"
        return self.df.index

    @property
    def start_time(self) -> pd.Timestamp:
        """First time instance (as pd.Timestamp)"""
        return self.time[0]

    @property
    def end_time(self) -> pd.Timestamp:
        """Last time instance (as pd.Timestamp)"""
        return self.time[-1]

    @property
    def values(self) -> np.ndarray:
        "Observed values"
        return self.df.values

    @property
    def n_points(self):
        """Number of observations"""
        return len(self.df)

    @property
    def filename(self):
        """Filename of the observation input"""
        return self._filename

    def _unit_text(self):
        override_units=self.override_units
        if self.itemInfo is None:
            return ""
        txt = f"{self.itemInfo.type.display_name}"
        if self.itemInfo.type != mikeio.EUMType.Undefined:
            if override_units==None:
                unit = self.itemInfo.unit.display_name
                txt = f"{txt} [{unit_display_name(unit)}]"
            else:
                unit = override_units
                txt = f"{txt} [{override_units}]"
        return txt

    def hist(self, bins=100, title=None, color=None, **kwargs):
        """plot histogram of observation values

        Wraps pandas.DataFrame hist() method.

        Parameters
        ----------
        bins : int, optional
            specification of bins, by default 100
        title : str, optional
            plot title, default: observation name
        color : str, optional
            plot color, by default "#d62728"
        kwargs : other keyword arguments to df.hist()

        Returns
        -------
        matplotlib axes
        """
        title = self.name if title is None else title
        kwargs["color"] = self.color if color is None else color

        ax = self.df.iloc[:, -1].hist(bins=bins, **kwargs)
        ax.set_title(title)
        ax.set_xlabel(self._unit_text())
        return ax

    def __copy__(self):
        return deepcopy(self)

    def copy(self):
        return self.__copy__()


class PointObservation(Observation):
    """Class for observations of fixed locations

    Create a PointObservation from a dfs0 file or a pd.DataFrame.

    Parameters
    ----------
    input : (str, pd.DataFrame, pd.Series)
        dfs0 filename or dataframe with the data
    item : (int, str), optional
        index or name of the wanted item, by default None
    x : float, optional
        x-coordinate of the observation point, by default None
    y : float, optional
        y-coordinate of the observation point, by default None
    z : float, optional
        z-coordinate of the observation point, by default None
    name : str, optional
        user-defined name for easy identification in plots etc, by default file basename
    variable_name : str, optional
        user-defined variable name in case of multiple variables, by default eumType name
    units : str, optional
        user-defined units name in case user wants to override eumUnits 

    Examples
    --------
    >>> o1 = PointObservation("klagshamn.dfs0", item=0, x=366844, y=6154291, name="Klagshamn")
    >>> o1 = PointObservation("klagshamn.dfs0", item="Water Level", x=366844, y=6154291)
    >>> o1 = PointObservation(df, item=0, x=366844, y=6154291, name="Klagshamn")
    >>> o1 = PointObservation(df["Water Level"], x=366844, y=6154291)
    """

    @property
    def geometry(self):
        from shapely.geometry import Point

        """Coordinates of observation"""
        if self.z is None:
            return Point(self.x, self.y)
        else:
            return Point(self.x, self.y, self.z)

    def __init__(
        self,
        filename,
        *,
        item=None,
        x: float = None,
        y: float = None,
        z: float = None,
        name: str = None,
        variable_name: str = None,
        units: str = None,
    ):

        self.x = x
        self.y = y
        self.z = z

        self._filename = None
        self._item = None

        if isinstance(filename, pd.Series):
            df = filename.to_frame()
            if name is None:
                name = "Observation"
            itemInfo = mikeio.ItemInfo(mikeio.EUMType.Undefined)
        elif isinstance(filename, pd.DataFrame):
            df = filename
            default_name = "Observation"
            if item is None:
                if len(df.columns) == 1:
                    item = 0
                else:
                    raise ValueError(
                        "item needs to be specified (more than one column in dataframe)"
                    )
            self._item = item

            if isinstance(item, str):
                df = df[[item]]
                default_name = item
            elif isinstance(item, int):
                if item < 0:
                    item = len(df.columns) + item
                default_name = df.columns[item]
                df = df.iloc[:, item]
            else:
                raise TypeError("item must be int or string")
            if name is None:
                name = default_name
            itemInfo = mikeio.ItemInfo(mikeio.EUMType.Undefined)
        elif isinstance(filename, str):
            assert os.path.exists(filename)
            self._filename = filename
            if name is None:
                name = os.path.basename(filename).split(".")[0]

            ext = os.path.splitext(filename)[-1]
            if ext == ".dfs0":
                df, itemInfo = self._read_dfs0(mikeio.open(filename), item)
                self._item = itemInfo.name
            else:
                raise NotImplementedError("Only dfs0 files supported")
        else:
            raise TypeError(
                f"input must be str, pandas Series/DataFrame! Given input has type {type(filename)}"
            )

        if not df.index.is_unique:
            # TODO: duplicates_keep="mean","first","last"
            raise ValueError(
                "Time axis has duplicate entries. It must be monotonically increasing."
            )

        super().__init__(
            name=name, df=df, itemInfo=itemInfo, variable_name=variable_name, override_units=units,
        )

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
        if item is None:
            if len(dfs.items) == 1:
                item = 0
            else:
                raise ValueError("item needs to be specified (more than one in file)")
        ds = dfs.read(items=item)
        itemInfo = ds.items[0]
        df = ds.to_dataframe()
        df.dropna(inplace=True)
        return df, itemInfo

    def plot(self, title=None, color=None, marker=".", linestyle="None", **kwargs):
        """plot observation timeseries

        Wraps pandas.DataFrame plot() method.

        Parameters
        ----------
        title : str, optional
            plot title, default: [name]
        color : str, optional
            plot color, by default '#d62728'
        marker : str, optional
            plot marker, by default '.'
        linestyle : str, optional
            line style, by default None
        kwargs: other keyword arguments to df.plot()
        """
        kwargs["color"] = self.color if color is None else color
        ax = self.df.plot(marker=marker, linestyle=linestyle, **kwargs)

        title = self.name if title is None else title
        ax.set_title(title)

        ax.set_ylabel(self._unit_text())
        return ax


class TrackObservation(Observation):
    """Class for observation with locations moving in space, e.g. satellite altimetry

    The data needs in addition to the datetime of each single observation point also, x and y coordinates.

    Create TrackObservation from dfs0 or DataFrame

    Parameters
    ----------
    input : (str, pd.DataFrame)
        path to dfs0 file or DataFrame with track data
    item : (str, int), optional
        item name or index of values, by default 2
    name : str, optional
        user-defined name for easy identification in plots etc, by default file basename
    variable_name : str, optional
        user-defined variable name in case of multiple variables, by default eumType name
    x_item : (str, int), optional
        item name or index of x-coordinate, by default 0
    y_item : (str, int), optional
        item name or index of y-coordinate, by default 1
    offset_duplicates : float, optional
        in case of duplicate timestamps, add this many seconds to consecutive duplicate entries, by default 0.001
    units : str, optional
        user-defined units name in case user wants to override eumUnits 


    Examples
    --------
    >>> o1 = TrackObservation("track.dfs0", item=2, name="c2")

    >>> o1 = TrackObservation("track.dfs0", item="wind_speed", name="c2")

    >>> o1 = TrackObservation("lon_after_lat.dfs0", item="wl", x_item=1, y_item=0)

    >>> o1 = TrackObservation("track_wl.dfs0", item="wl", x_item="lon", y_item="lat")

    >>> df = pd.DataFrame(
    ...         {
    ...             "t": pd.date_range("2010-01-01", freq="10s", periods=n),
    ...             "x": np.linspace(0, 10, n),
    ...             "y": np.linspace(45000, 45100, n),
    ...             "swh": [0.1, 0.3, 0.4, 0.5, 0.3],
    ...         }
    ... )
    >>> df = df.set_index("t")
    >>> df
                        x        y  swh
    t
    2010-01-01 00:00:00   0.0  45000.0  0.1
    2010-01-01 00:00:10   2.5  45025.0  0.3
    2010-01-01 00:00:20   5.0  45050.0  0.4
    2010-01-01 00:00:30   7.5  45075.0  0.5
    2010-01-01 00:00:40  10.0  45100.0  0.3
    >>> t1 = TrackObservation(df, name="fake")
    >>> t1.n_points
    5
    >>> t1.values
    array([0.1, 0.3, 0.4, 0.5, 0.3])
    >>> t1.time
    DatetimeIndex(['2010-01-01 00:00:00', '2010-01-01 00:00:10',
               '2010-01-01 00:00:20', '2010-01-01 00:00:30',
               '2010-01-01 00:00:40'],
              dtype='datetime64[ns]', name='t', freq=None)
    >>> t1.x
    array([ 0. ,  2.5,  5. ,  7.5, 10. ])
    >>> t1.y
    array([45000., 45025., 45050., 45075., 45100.])

    """

    @property
    def geometry(self):
        from shapely.geometry import MultiPoint

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

    def __init__(
        self,
        input,
        *,
        item: int = None,
        name: str = None,
        variable_name: str = None,
        x_item=0,
        y_item=1,
        offset_duplicates: float = 0.001,
        units: str = None,
    ):

        self._filename = None
        self._item = None

        if isinstance(input, pd.DataFrame):
            df = input
            df_items = df.columns.to_list()
            items = self._parse_track_items(df_items, x_item, y_item, item)
            df = df.iloc[:, items].copy()
            itemInfo = mikeio.ItemInfo(mikeio.EUMType.Undefined)
        elif isinstance(input, str):
            assert os.path.exists(input)
            self._filename = input
            if name is None:
                name = os.path.basename(input).split(".")[0]

            ext = os.path.splitext(input)[-1]
            if ext == ".dfs0":
                dfs = mikeio.open(input)
                file_items = [i.name for i in dfs.items]
                items = self._parse_track_items(file_items, x_item, y_item, item)
                df, itemInfo = self._read_dfs0(dfs, items)
                self._item = itemInfo.name
            else:
                raise NotImplementedError(
                    "Only dfs0 files and DataFrames are supported"
                )
        else:
            raise TypeError(
                f"input must be str or pandas DataFrame! Given input has type {type(input)}"
            )

        # A unique index makes lookup much faster O(1)
        if not df.index.is_unique:
            df.index = make_unique_index(df.index, offset_duplicates=offset_duplicates)

        super().__init__(
            name=name, df=df, itemInfo=itemInfo, variable_name=variable_name,override_units=units,
        )

    @staticmethod
    def _parse_track_items(items, x_item, y_item, item):
        """If input has exactly 3 items we accept item=None"""
        if len(items) < 3:
            raise ValueError(
                f"Input has only {len(items)} items. It should have at least 3."
            )
        if item is None:
            if len(items) == 3:
                item = 2
            elif len(items) > 3:
                raise ValueError("Input has more than 3 items, but item was not given!")
        else:
            item = _parse_item(items, item)

        x_item = _parse_item(items, x_item, "x_item")
        y_item = _parse_item(items, y_item, "y_item")

        if (item == x_item) or (item == y_item) or (x_item == y_item):
            raise ValueError(
                f"x-item ({x_item}), y-item ({y_item}) and value-item ({item}) must be different!"
            )
        return [x_item, y_item, item]

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

    res = name.replace("meter", "m").replace("_per_", "/").replace(" per ", "/").replace("second", "s").replace("sec", "s")

    return res
