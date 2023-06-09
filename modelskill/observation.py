"""
The `observation` module contains different types of Observation classes for
fixed locations (PointObservation), or locations moving in space (TrackObservation).

Examples
--------
>>> o1 = PointObservation("klagshamn.dfs0", item=0, x=366844, y=6154291, name="Klagshamn")
"""
import os
from typing import Optional, Union
import numpy as np
import pandas as pd
import mikeio
from copy import deepcopy

from .utils import make_unique_index
from .types import Quantity
from .timeseries import TimeSeries


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


class Observation(TimeSeries):
    """Base class for observations

    Parameters
    ----------
    data : pd.DataFrame
    name : str, optional
        user-defined name, e.g. "Station A", by default "Observation"
    quantity : Optional[Quantity], optional
        The quantity of the observation, for validation with model results
    weight : float, optional
        weighting factor, to be used in weighted skill scores, by default 1.0
    color : str, optional
        color of the observation in plots, by default "#d62728"
    """

    def __init__(
        self,
        data: pd.DataFrame,
        name: str = "Observation",
        quantity: Optional[Quantity] = None,
        weight: float = 1.0,
        color: str = "#d62728",
    ):

        if name is None:
            name = "Observation"

        # TODO move this to TimeSeries?
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError(
                f"Input must have a datetime index! Provided index was {type(data.index)}"
            )
        time = data.index.round(freq="100us")  # 0.0001s accuracy
        data.index = pd.DatetimeIndex(time, freq="infer")

        self.weight = weight

        super().__init__(name=name, data=data, quantity=quantity, color=color)

    @property
    def values(self) -> np.ndarray:
        "Observed values"
        return self.data.values

    @property
    def n_points(self):
        """Number of observations"""
        return len(self.data)

    def copy(self):
        return deepcopy(self)


class PointObservation(Observation):
    """Class for observations of fixed locations

    Create a PointObservation from a dfs0 file or a pd.DataFrame.

    Parameters
    ----------
    data : (str, pd.DataFrame, pd.Series)
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
    quantity : Quantity, optional
        The quantity of the observation, for validation with model results

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

        if self.z is None:
            return Point(self.x, self.y)
        else:
            return Point(self.x, self.y, self.z)

    def __init__(
        self,
        data,
        *,
        item=None,
        x: float = None,
        y: float = None,
        z: float = None,
        name: str = None,
        quantity: Optional[Union[str, Quantity]] = None,
    ):

        self.x = x
        self.y = y
        self.z = z

        self._filename = None
        self._item = None

        # TODO move this to TimeSeries?
        if isinstance(data, pd.Series):
            df = data.to_frame()
            if name is None:
                name = "Observation"
        elif isinstance(data, mikeio.DataArray):
            df = data.to_dataframe()
            if quantity is None:
                quantity = Quantity.from_mikeio_iteminfo(data.item)
        elif isinstance(data, mikeio.Dataset):
            df = data.to_dataframe()[[item]]
            if quantity is None:
                quantity = Quantity.from_mikeio_iteminfo(data[item].item)
        elif isinstance(data, pd.DataFrame):
            df = data
            default_name = "Observation"
            if item is None:
                if len(df.columns) == 1:
                    item = 0
                else:
                    raise ValueError(
                        f"item must be specified (more than one column in dataframe). Available columns: {list(df.columns)}"
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
        elif isinstance(data, str):
            assert os.path.exists(data)
            self._filename = data
            if name is None:
                name = os.path.basename(data).split(".")[0]

            ext = os.path.splitext(data)[-1]
            if ext == ".dfs0":
                df, iteminfo = self._read_dfs0(mikeio.open(data), item)
                if quantity is None:
                    quantity = Quantity.from_mikeio_iteminfo(iteminfo)
            else:
                raise NotImplementedError("Only dfs0 files supported")
        else:
            raise TypeError(
                f"input must be str, mikeio.DataArray/Dataset or pandas Series/DataFrame! Given input has type {type(data)}"
            )

        if not df.index.is_unique:
            # TODO: duplicates_keep="mean","first","last"
            raise ValueError(
                "Time axis has duplicate entries. It must be monotonically increasing."
            )

        super().__init__(
            name=name,
            data=df,
            quantity=quantity,
        )

    def __repr__(self):
        out = f"PointObservation: {self.name}, x={self.x}, y={self.y}"
        return out

    # TODO does this belong here?
    @staticmethod
    def _read_dfs0(dfs, item):
        """Read data from dfs0 file"""
        if item is None:
            if len(dfs.items) == 1:
                item = 0
            else:
                item_names = [i.name for i in dfs.items]
                raise ValueError(
                    f"item needs to be specified (more than one in file). Available items: {item_names} "
                )
        ds = dfs.read(items=item)
        itemInfo = ds.items[0]
        df = ds.to_dataframe()
        df.dropna(inplace=True)
        return df, itemInfo


class TrackObservation(Observation):
    """Class for observation with locations moving in space, e.g. satellite altimetry

    The data needs in addition to the datetime of each single observation point also, x and y coordinates.

    Create TrackObservation from dfs0 or DataFrame

    Parameters
    ----------
    data : (str, pd.DataFrame)
        path to dfs0 file or DataFrame with track data
    item : (str, int), optional
        item name or index of values, by default 2
    name : str, optional
        user-defined name for easy identification in plots etc, by default file basename
    x_item : (str, int), optional
        item name or index of x-coordinate, by default 0
    y_item : (str, int), optional
        item name or index of y-coordinate, by default 1
    offset_duplicates : float, optional
        in case of duplicate timestamps, add this many seconds to consecutive duplicate entries, by default 0.001


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
        return MultiPoint(self.data.iloc[:, 0:2].values)

    @property
    def x(self):
        return self.data.iloc[:, 0].values

    @property
    def y(self):
        return self.data.iloc[:, 1].values

    @property
    def values(self):
        return self.data.iloc[:, 2].values

    def __init__(
        self,
        data,
        *,
        item: int = None,
        name: str = None,
        x_item=0,
        y_item=1,
        offset_duplicates: float = 0.001,
        quantity: Optional[Quantity] = None,
    ):

        self._filename = None
        self._item = None

        if isinstance(data, pd.DataFrame):
            df = data
            df_items = df.columns.to_list()
            items = self._parse_track_items(df_items, x_item, y_item, item)
            df = df.iloc[:, items].copy()
        elif isinstance(data, str):
            assert os.path.exists(data)
            self._filename = data
            if name is None:
                name = os.path.basename(data).split(".")[0]

            ext = os.path.splitext(data)[-1]
            if ext == ".dfs0":
                dfs = mikeio.open(data)
                file_items = [i.name for i in dfs.items]
                items = self._parse_track_items(file_items, x_item, y_item, item)
                df, iteminfo = self._read_dfs0(dfs, items)
                if quantity is None:
                    quantity = Quantity.from_mikeio_iteminfo(iteminfo)
            else:
                raise NotImplementedError(
                    "Only dfs0 files and DataFrames are supported"
                )
        else:
            raise TypeError(
                f"input must be str or pandas DataFrame! Given input has type {type(data)}"
            )

        # A unique index makes lookup much faster O(1)
        if not df.index.is_unique:
            df.index = make_unique_index(df.index, offset_duplicates=offset_duplicates)

        # TODO is this needed elsewhere?
        # make sure location columns are named x and y
        if isinstance(x_item, str):
            old_x_name = x_item
        else:
            old_x_name = df.columns[x_item]

        if isinstance(y_item, str):
            old_y_name = y_item
        else:
            old_y_name = df.columns[y_item]

        df = df.rename(columns={old_x_name: "x", old_y_name: "y"})

        super().__init__(
            name=name,
            data=df,
            quantity=quantity,
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

    res = (
        name.replace("meter", "m")
        .replace("_per_", "/")
        .replace(" per ", "/")
        .replace("second", "s")
        .replace("sec", "s")
        .replace("degree", "°")
    )

    return res
