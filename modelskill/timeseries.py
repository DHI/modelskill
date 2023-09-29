from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Protocol
import numpy as np

import pandas as pd
import xarray as xr

from .types import GeometryType, Quantity

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
        )
        fig.show()


@dataclass
class TimeSeries:
    """Time series data"""

    data: xr.Dataset
    plotter: ClassVar = MatplotlibTimeSeriesPlotter  # TODO is this the best option to choose a plotter? Can we use the settings module?

    def __post_init__(self) -> None:
        self.data = self._validate_data(self.data)
        self.plot: TimeSeriesPlotter = TimeSeries.plotter(self)
        self.hist = self.plot.hist  # TODO remove this

    @staticmethod
    def _validate_data(ds) -> xr.Dataset:
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
        assert "x" in ds, "data must have an x-coordinate"
        assert "y" in ds, "data must have a y-coordinate"
        # assert "z" in ds, "data must have a z-coordinate"

        # Validate data
        vars = [v for v in ds.data_vars if v != "x" and v != "y" and v != "z"]
        assert len(vars) > 0, "data must have at least one item"
        # assert len(ds["time"]) > 0, "data must have at least one time"
        for v in vars:
            assert (
                len(ds[v].dims) == 1
            ), f"Only 0-dimensional data arrays are supported! {v} has {len(ds[v].dims)} dimensions"
            assert (
                list(ds[v].dims)[0] == "time"
            ), f"All data arrays must have a time dimension; {v} has dimensions {ds[v].dims}"

        # Validate primary data array
        name = str(vars[0])
        da = ds[name]  # By definition, first data variable is the value variable!
        assert (
            "kind" in da.attrs
        ), f"The first data array {vars[0]} (the value array) must have a kind attribute!"
        assert da.attrs["kind"] in [
            "model",
            "observation",
        ], f"data array {da} attribute 'kind' must be 'model' or 'observation', not {da.attrs['kind']}"

        # Validate aux data arrays
        for v in vars[1:]:
            if "kind" in ds[v].attrs:
                assert ds[v].attrs["kind"] not in [
                    "model",
                    "observation",
                ], f"data can only have one model/observation array! {vars[0]} is the first array and by definition the 'value' array, any subsequent arrays must be of kind 'aux', but array {v} has kind {ds[v].attrs['kind']}!"

        # Validate attrs
        assert "gtype" in ds.attrs, "data must have a gtype attribute"
        if "long_name" not in da.attrs:
            da.attrs["long_name"] = Quantity.undefined().name

        if "units" not in da.attrs:
            da.attrs["units"] = Quantity.undefined().unit
        # assert "quantity" in da.attrs, "data must have a quantity attribute"
        # assert "name" in da.attrs["quantity"]
        # assert "unit" in da.attrs["quantity"]
        color = da.attrs["color"] if "color" in da.attrs else None
        da.attrs["color"] = TimeSeries._parse_color(name, color=color)

        return ds

    @staticmethod
    def _validate_name(name: str) -> str:
        """Validate name"""
        assert isinstance(name, str), "name must be a string"
        RESERVED_NAMES = ["x", "y", "z", "time"]
        assert (
            name not in RESERVED_NAMES
        ), f"name '{name}' is reserved and cannot be used! Please choose another name."
        return name

    @property
    def name(self) -> str:
        """Name of time series (value item name)"""
        return self._val_item

    # setter
    @name.setter
    def name(self, name: str) -> None:
        self.data = self.data.rename({self._val_item: name})

    @property
    def quantity(self) -> Quantity:
        """Quantity of time series"""
        return Quantity(
            name=self.data[self._val_item].attrs["long_name"],
            unit=self.data[self._val_item].attrs["units"],
        )

    @quantity.setter
    def quantity(self, quantity: Quantity) -> None:
        self.data[self._val_item].attrs["long_name"] = quantity.name
        self.data[self._val_item].attrs["units"] = quantity.unit

    @property
    def color(self) -> str:
        """Color of time series"""
        return self.data[self._val_item].attrs["color"]

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
    def _val_item(self) -> str:
        # TODO: better way to find the value item
        # (when aux is introduced this will fail need fixing)
        vars = [v for v in self.data.data_vars if v != "x" and v != "y" and v != "z"]
        return str(vars[0])

    @property
    def values(self) -> np.ndarray:
        """Values as numpy array"""
        return self.data[self._val_item].values

    @property
    def _values_as_series(self) -> pd.Series:
        """Values to series (for plotting)"""
        return self.data[self._val_item].to_series()

    @property
    def start_time(self) -> pd.Timestamp:
        """Start time of time series data"""
        return self.time[0]  # type: ignore

    @property
    def end_time(self) -> pd.Timestamp:
        """End time of time series data"""
        return self.time[-1]  # type: ignore

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}> '{self.name}'"

    # len() of a DataFrame returns the number of rows,
    # len() of xr.Dataset returns the number of variables
    # what should len() of TimeSeries return?
    def __len__(self) -> int:
        return len(self.data.time)

    @property
    def n_points(self):
        """Number of data points"""
        return len(self.data.time)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        if self.gtype == str(GeometryType.POINT):
            # we need to remove the scalar coordinate variables as they
            # will otherwise be columns in the dataframe
            return self.data.drop_vars(["x", "y", "z"])[self.name].to_dataframe()
        else:
            return self.data[["x", "y", self.name]].to_dataframe()

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
