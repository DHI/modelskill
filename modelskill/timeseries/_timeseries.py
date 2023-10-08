from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
import pandas as pd
import xarray as xr

from ..types import GeometryType, Quantity
from ._plotter import TimeSeriesPlotter, MatplotlibTimeSeriesPlotter

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


def _validate_name(name: str) -> str:
    assert isinstance(name, str), "name must be a string"
    RESERVED_NAMES = ["x", "y", "z", "time"]
    assert (
        name not in RESERVED_NAMES
    ), f"name '{name}' is reserved and cannot be used! Please choose another name."
    return name


def _parse_color(name: str, color: str | None = None) -> str:
    from matplotlib.colors import is_color_like

    if color is None:
        idx = hash(name) % len(DEFAULT_COLORS)
        color = DEFAULT_COLORS[idx]
    if not is_color_like(color):
        raise ValueError(f"color must be a valid (matplotlib) color, not {color}")
    return color


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
    da.attrs["color"] = _parse_color(name, color=color)

    return ds


@dataclass
class TimeSeries:
    """Time series data"""

    data: xr.Dataset
    plotter: ClassVar = MatplotlibTimeSeriesPlotter  # TODO is this the best option to choose a plotter? Can we use the settings module?

    def __post_init__(self) -> None:
        self.data = _validate_dataset(self.data)
        self.plot: TimeSeriesPlotter = TimeSeries.plotter(self)

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
        self.data[self.name].attrs["color"] = _parse_color(self.name, color)

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
        return f"<{self.__class__.__name__}> '{self.name}' (n_points: {self.n_points})"

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

        self.data = self.data.sel(time=slice(start_time, end_time))

    def interp_time(self, new_time: pd.DatetimeIndex) -> TimeSeries:
        """Interpolate time series to new time index

        Parameters
        ----------
        new_time : pd.DatetimeIndex
            new time index

        Returns
        -------
        TimeSeries
            interpolated time series
        """
        if not isinstance(new_time, pd.DatetimeIndex):
            try:
                new_time = pd.DatetimeIndex(new_time)
            except Exception:
                raise ValueError(
                    "new_time must be a pandas DatetimeIndex (or convertible to one)"
                )

        # TODO: self.__class__
        return TimeSeries(self.data.interp(time=new_time))