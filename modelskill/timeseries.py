from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol, get_args, Optional, List
import numpy as np
import pandas as pd
import xarray as xr

import mikeio

from .types import GeometryType, PointType, Quantity, TrackType
from .utils import _get_name, make_unique_index, get_item_name_and_idx

DEFAULT_COLORS = [
    "#b30000",
    "#7c1158",
    "#4421af",
    "#1a53ff",
    "#0d88e6",
    "#00b7c7",
    "#5ad45a",
    "#8be04e",
    "#ebdc78",
]


def _parse_point_input(
    data: PointType,
    name: Optional[str],
    item: str | int | None,
    quantity: Optional[Quantity],
) -> xr.Dataset:
    """Convert accepted input data to an xr.Dataset"""
    assert isinstance(
        data, get_args(PointType)
    ), f"Could not construct object from provided data of type {type(data)}"

    if isinstance(data, (str, Path)):
        assert Path(data).suffix == ".dfs0", "File must be a dfs0 file"
        name = name or Path(data).stem
        data = mikeio.read(data)  # now mikeio.Dataset
    elif isinstance(data, mikeio.Dfs0):
        data = data.read()  # now mikeio.Dataset

    # parse items
    if isinstance(data, (mikeio.DataArray, pd.Series, xr.DataArray)):  # type: ignore
        item_name = data.name if hasattr(data, "name") else "PointModelResult"
        if item is not None:
            raise ValueError(f"item must be None when data is a {type(data)}")
    elif isinstance(data, (mikeio.Dataset, pd.DataFrame, xr.Dataset)):  # type: ignore
        if isinstance(data, mikeio.Dataset):
            valid_items = [i.name for i in data.items]
        elif isinstance(data, pd.DataFrame):
            valid_items = list(data.columns)
        else:
            valid_items = list(data.data_vars)
        item_name = _get_name(x=item, valid_names=valid_items)

    # select relevant items
    if isinstance(data, mikeio.DataArray):
        data = mikeio.Dataset(data)
    elif isinstance(data, pd.Series):
        data = data.to_frame()
    elif isinstance(data, xr.DataArray):
        data = data.to_dataset()
    else:
        data = data[[item_name]]
    assert isinstance(data, (mikeio.Dataset, pd.DataFrame, xr.Dataset))

    # parse quantity
    if isinstance(data, mikeio.Dataset):
        if quantity is None:
            quantity = Quantity.from_mikeio_iteminfo(data[0].item)
    model_quantity = Quantity.undefined() if quantity is None else quantity

    # convert to xr.Dataset
    if isinstance(data, mikeio.Dataset):
        ds = data.to_xarray()
    elif isinstance(data, pd.DataFrame):
        data.index.name = "time"
        ds = data.to_xarray()
    else:
        assert len(data.dims) == 1, "Only 0-dimensional data are supported"
        if data.coords[list(data.coords)[0]].name != "time":
            data = data.rename({list(data.coords)[0]: "time"})
        ds = data

    name = name or item_name
    name = _validate_name(name)

    # basic processing
    ds = ds.dropna(dim="time")

    vars = [v for v in ds.data_vars]
    assert len(ds.data_vars) == 1
    ds = ds.rename({vars[0]: name})

    ds[name].attrs["long_name"] = model_quantity.name
    ds[name].attrs["units"] = model_quantity.unit

    ds.attrs["gtype"] = str(GeometryType.POINT)
    return ds


@dataclass
class TrackItem:
    x: str
    y: str
    values: str

    @property
    def all(self) -> List[str]:
        return [self.x, self.y, self.values]


def _parse_track_items(items, x_item, y_item, item) -> TrackItem:
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

    item, _ = get_item_name_and_idx(items, item)
    x_item, _ = get_item_name_and_idx(items, x_item)
    y_item, _ = get_item_name_and_idx(items, y_item)

    if (item == x_item) or (item == y_item) or (x_item == y_item):
        raise ValueError(
            f"x-item ({x_item}), y-item ({y_item}) and value-item ({item}) must be different!"
        )
    return TrackItem(x=x_item, y=y_item, values=item)


def _parse_track_input(
    data: TrackType,
    name: Optional[str],
    item: str | int | None,
    quantity: Optional[Quantity],
    x_item: str | int,
    y_item: str | int,
) -> xr.Dataset:
    assert isinstance(
        data, get_args(TrackType)
    ), "Could not construct TrackModelResult from provided data."

    if isinstance(data, (str, Path)):
        assert Path(data).suffix == ".dfs0", "File must be a dfs0 file"
        name = name or Path(data).stem
        data = mikeio.read(data)  # now mikeio.Dataset
    elif isinstance(data, mikeio.Dfs0):
        data = data.read()  # now mikeio.Dataset

    # parse items
    if isinstance(data, mikeio.Dataset):
        valid_items = [i.name for i in data.items]
    elif isinstance(data, pd.DataFrame):
        valid_items = list(data.columns)
    elif isinstance(data, xr.Dataset):
        valid_items = list(data.data_vars)
    else:
        raise ValueError("Could not construct TrackModelResult from provided data")

    ti = _parse_track_items(valid_items, x_item, y_item, item)
    name = name or ti.values
    name = _validate_name(name)

    # parse quantity
    if isinstance(data, mikeio.Dataset):
        if quantity is None:
            quantity = Quantity.from_mikeio_iteminfo(data[ti.values].item)
    model_quantity = Quantity.undefined() if quantity is None else quantity

    # select relevant items and convert to xr.Dataset
    assert isinstance(data, (mikeio.Dataset, pd.DataFrame, xr.Dataset))
    data = data[ti.all]
    if isinstance(data, mikeio.Dataset):
        ds = data.to_xarray()
    elif isinstance(data, pd.DataFrame):
        data.index.name = "time"
        ds = data.to_xarray()
    else:
        assert len(data.dims) == 1, "Only 0-dimensional data are supported"
        if data.coords[list(data.coords)[0]].name != "time":
            data = data.rename({list(data.coords)[0]: "time"})
        ds = data

    ds = ds.rename({ti.x: "x", ti.y: "y"})
    ds = ds.dropna(dim="time", subset=["x", "y"])
    ds["time"] = make_unique_index(ds["time"].to_index(), offset_duplicates=0.001)

    SPATIAL_DIMS = ["x", "y", "z"]

    for dim in SPATIAL_DIMS:
        if dim in ds:
            ds = ds.set_coords(dim)

    assert len(ds.data_vars) == 1
    data_var = str(list(ds.data_vars)[0])
    ds = ds.rename({data_var: name})
    ds[name].attrs["long_name"] = model_quantity.name
    ds[name].attrs["units"] = model_quantity.unit

    ds.attrs["gtype"] = str(GeometryType.TRACK)
    return ds


def _validate_name(name: str) -> str:
    assert isinstance(name, str), "name must be a string"
    RESERVED_NAMES = ["x", "y", "z", "time"]
    assert (
        name not in RESERVED_NAMES
    ), f"name '{name}' is reserved and cannot be used! Please choose another name."
    return name


class TimeSeriesPlotter(Protocol):
    def __init__(self, ts: "TimeSeries") -> None:
        pass

    def __call__(self):
        pass

    def plot(self):
        pass

    def hist(self):
        pass

    def timeseries(self):
        pass


class MatplotlibTimeSeriesPlotter(TimeSeriesPlotter):
    def __init__(self, ts: "TimeSeries") -> None:
        self._ts = ts

    def __call__(self, title=None, color=None, marker=".", linestyle="None", **kwargs):
        self.plot(
            title=title, color=color, marker=marker, linestyle=linestyle, **kwargs
        )

    def timeseries(
        self, title=None, color=None, marker=".", linestyle="None", **kwargs
    ):
        """plot timeseries

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
        self.plot(
            title=title, color=color, marker=marker, linestyle=linestyle, **kwargs
        )

    def plot(self, title=None, color=None, marker=".", linestyle="None", **kwargs):
        """plot timeseries

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
        kwargs["color"] = self._ts.color if color is None else color
        ax = self._ts._values_as_series.plot(
            marker=marker, linestyle=linestyle, **kwargs
        )

        title = self._ts.name if title is None else title
        ax.set_title(title)

        ax.set_ylabel(str(self._ts.quantity))
        return ax

    def hist(self, bins=100, title=None, color=None, **kwargs):
        """plot histogram of timeseries values

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
        title = self._ts.name if title is None else title

        kwargs["color"] = self._ts.color if color is None else color

        ax = self._ts._values_as_series.hist(bins=bins, **kwargs)
        ax.set_title(title)
        ax.set_xlabel(str(self._ts.quantity))
        return ax


class PlotlyTimeSeriesPlotter(TimeSeriesPlotter):
    def __init__(self, ts: "TimeSeries") -> None:
        self._ts: "TimeSeries" = ts

    def __call__(self):
        self.plot()

    def timeseries(self):
        self.plot()

    def plot(self):
        import plotly.express as px  # type: ignore

        fig = px.line(
            self._ts._values_as_series, color_discrete_sequence=[self._ts.color]
        )
        fig.show()

    def hist(self, bins=100, **kwargs):
        import plotly.express as px  # type: ignore

        fig = px.histogram(
            self._ts._values_as_series,
            nbins=bins,
            color_discrete_sequence=[self._ts.color],
            **kwargs,
        )
        fig.show()


@dataclass
class TimeSeries:
    """Time series data"""

    data: xr.Dataset
    plotter: ClassVar = MatplotlibTimeSeriesPlotter  # TODO is this the best option to choose a plotter? Can we use the settings module?

    def __post_init__(self) -> None:
        self.data = self._validate_dataset(self.data)
        self.plot: TimeSeriesPlotter = TimeSeries.plotter(self)
        self.hist = self.plot.hist  # TODO remove this

    @staticmethod
    def _validate_dataset(ds) -> xr.Dataset:
        """Validate data"""
        assert isinstance(ds, xr.Dataset), "data must be a xr.Dataset"

        # Validate time
        assert len(ds.dims) == 1, "Only 0-dimensional data are supported"
        assert list(ds.dims)[0] == "time", "data must have a time dimension"
        assert isinstance(ds.time.to_index(), pd.DatetimeIndex), "time must be datetime"
        assert (
            ds.time.to_index().is_monotonic_increasing
        ), "time must be increasing (please check for duplicate times))"

        # Validate coordinates
        assert "x" in ds.coords, "data must have an x-coordinate"
        assert "y" in ds.coords, "data must have a y-coordinate"
        if "z" not in ds.coords:
            ds.coords["z"] = np.nan
        # assert "z" in ds.coords, "data must have a z-coordinate"

        # Validate data
        vars = [v for v in ds.data_vars]
        assert len(vars) > 0, "data must have at least one data array"
        # assert len(ds["time"]) > 0, "data must have at least one time"
        name = ""
        n_primary = 0
        for v in vars:
            v = _validate_name(str(v))
            assert (
                len(ds[v].dims) == 1
            ), f"Only 0-dimensional data arrays are supported! {v} has {len(ds[v].dims)} dimensions"
            assert (
                list(ds[v].dims)[0] == "time"
            ), f"All data arrays must have a time dimension; {v} has dimensions {ds[v].dims}"
            if "kind" not in ds[v].attrs:
                ds[v].attrs["kind"] = "auxiliary"
            if ds[v].attrs["kind"] in ["model", "observation"]:
                n_primary += 1
                name = v

        # Validate primary data array
        if n_primary == 0:
            raise ValueError(
                "data must have at least one model or observation array (marked by the kind attribute)"
            )
        if n_primary > 1:
            raise ValueError(
                "data can only have one model or observation array (marked by the kind attribute)"
            )
        da = ds[name]

        # Validate attrs
        assert "gtype" in ds.attrs, "data must have a gtype attribute"
        assert ds.attrs["gtype"] in [
            str(GeometryType.POINT),
            str(GeometryType.TRACK),
        ], f"data attribute 'gtype' must be one of {GeometryType.POINT} or {GeometryType.TRACK}"
        if "long_name" not in da.attrs:
            da.attrs["long_name"] = Quantity.undefined().name
        if "units" not in da.attrs:
            da.attrs["units"] = Quantity.undefined().unit
        color = da.attrs["color"] if "color" in da.attrs else None
        da.attrs["color"] = TimeSeries._parse_color(name, color=color)

        return ds

    @property
    def _val_item(self) -> str:
        return [
            str(v)
            for v in self.data.data_vars
            if self.data[v].attrs["kind"] == "model"
            or self.data[v].attrs["kind"] == "observation"
        ][0]

    @property
    def name(self) -> str:
        """Name of time series (value item name)"""
        return self._val_item

    # setter
    @name.setter
    def name(self, name: str) -> None:
        name = _validate_name(name)
        self.data = self.data.rename({self._val_item: name})

    @property
    def quantity(self) -> Quantity:
        """Quantity of time series"""
        return Quantity(
            name=self.data[self.name].attrs["long_name"],
            unit=self.data[self.name].attrs["units"],
        )

    @quantity.setter
    def quantity(self, quantity: Quantity) -> None:
        assert isinstance(quantity, Quantity), "value must be a Quantity object"
        self.data[self.name].attrs["long_name"] = quantity.name
        self.data[self.name].attrs["units"] = quantity.unit

    @property
    def color(self) -> str:
        """Color of time series"""
        return self.data[self.name].attrs["color"]

    @color.setter
    def color(self, color: str | None) -> None:
        self.data[self.name].attrs["color"] = self._parse_color(self.name, color)

    @staticmethod
    def _parse_color(name: str, color: str | None = None) -> str:
        from matplotlib.colors import is_color_like

        if color is None:
            idx = hash(name) % len(DEFAULT_COLORS)
            color = DEFAULT_COLORS[idx]
        if not is_color_like(color):
            raise ValueError(f"color must be a valid (matplotlib) color, not {color}")
        return color

    @property
    def gtype(self):
        """Geometry type"""
        return self.data.attrs["gtype"]

    @property
    def time(self) -> pd.DatetimeIndex:
        """Time index"""
        return pd.DatetimeIndex(self.data.time)

    @property
    def x(self):
        """x-coordinate"""
        return self._coordinate_values("x")

    @x.setter
    def x(self, value):
        self.data["x"] = value

    @property
    def y(self):
        """y-coordinate"""
        return self._coordinate_values("y")

    @y.setter
    def y(self, value):
        self.data["y"] = value

    def _coordinate_values(self, coord):
        vals = self.data[coord].values
        return np.atleast_1d(vals)[0] if vals.ndim == 0 else vals

    @property
    def _is_modelresult(self) -> bool:
        return self.data[self.name].attrs["kind"] == "model"

    @property
    def values(self) -> np.ndarray:
        """Values as numpy array"""
        return self.data[self.name].values

    @property
    def _values_as_series(self) -> pd.Series:
        """Values to series (for plotting)"""
        return self.data[self.name].to_series()

    @property
    def start_time(self) -> pd.Timestamp:
        """Start time of time series data"""
        return self.time[0]  # type: ignore

    @property
    def end_time(self) -> pd.Timestamp:
        """End time of time series data"""
        return self.time[-1]  # type: ignore

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}> '{self.name}' (n_points: {self.n_points}))"

    # len() of a DataFrame returns the number of rows,
    # len() of xr.Dataset returns the number of variables
    # what should len() of TimeSeries return?
    def __len__(self) -> int:
        return len(self.data.time)

    @property
    def n_points(self):
        """Number of data points"""
        return len(self.data.time)

    def copy(self):
        return deepcopy(self)

    def equals(self, other: TimeSeries) -> bool:
        """Check if two TimeSeries are equal"""
        return self.data.equals(other.data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeSeries):
            raise NotImplementedError("Can only compare TimeSeries objects")
        return self.equals(other)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        if self.gtype == str(GeometryType.POINT):
            # we remove the scalar coordinate variables as they
            # will otherwise be columns in the dataframe
            return self.data.drop_vars(["x", "y", "z"])[self.name].to_dataframe()
        else:
            return self.data.drop_vars(["z"])[["x", "y", self.name]].to_dataframe()

    def trim(
        self, start_time: pd.Timestamp, end_time: pd.Timestamp, buffer="1s"
    ) -> None:
        """Trim observation data to a given time interval

        Parameters
        ----------
        start_time : pd.Timestamp
            start time
        end_time : pd.Timestamp
            end time
        buffer : str, optional
            buffer time around start and end time, by default "1s"
        """
        # Expand time interval with buffer
        start_time = pd.Timestamp(start_time) - pd.Timedelta(buffer)
        end_time = pd.Timestamp(end_time) + pd.Timedelta(buffer)

        if isinstance(self.data, (pd.DataFrame, pd.Series)):
            self.data = self.data.loc[start_time:end_time]  # type: ignore
        else:
            # assume xr
            self.data = self.data.sel(time=slice(start_time, end_time))


# TODO: add interp_time method
#     def interp_time(self, new_time: pd.DatetimeIndex) -> TimeSeries:
#         """Interpolate time series to new time index

#         Parameters
#         ----------
#         new_time : pd.DatetimeIndex
#             new time index

#         Returns
#         -------
#         TimeSeries
#             interpolated time series
#         """
#         new_df = _interp_time(self.data, new_time)
#         return TimeSeries(
#             name=self.name,
#             data=new_df,
#             quantity=self.quantity,
#             color=self.color,
#         )


# def _interp_time(df: pd.DataFrame, new_time: pd.DatetimeIndex) -> pd.DataFrame:
#     """Interpolate time series to new time index"""
#     new_df = (
#         df.reindex(df.index.union(new_time))
#         .interpolate(method="time", limit_area="inside")
#         .reindex(new_time)
#     )
#     return new_df
