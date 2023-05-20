from datetime import datetime
from dataclasses import dataclass
from typing import ClassVar, Protocol

import pandas as pd

from .types import Quantity


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
        self.plot()

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
        ax = self._ts.data.plot(marker=marker, linestyle=linestyle, **kwargs)

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

        ax = self._ts.data.iloc[:, -1].hist(bins=bins, **kwargs)
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

        fig = px.line(self._ts.data, color_discrete_sequence=[self._ts.color])
        fig.show()

    def hist(self, bins=100, **kwargs):
        import plotly.express as px  # type: ignore

        fig = px.histogram(
            self._ts.data, nbins=bins, color_discrete_sequence=[self._ts.color]
        )
        fig.show()


@dataclass
class TimeSeries:
    """Time series data"""

    plotter: ClassVar = MatplotlibTimeSeriesPlotter  # TODO is this the best option to choose a plotter? Can we use the settings module?

    name: str
    data: pd.DataFrame
    quantity: Quantity
    color: str = "#d62728"

    def __post_init__(self) -> None:
        # TODO is __post_init__ the most elegant way to handle this?
        if self.quantity is None:
            self.quantity = Quantity.undefined()
        elif isinstance(self.quantity, str):
            self.quantity = Quantity.from_mikeio_eum_name(self.quantity)

        assert isinstance(self.quantity, Quantity)

        assert isinstance(
            self.data, (pd.DataFrame, pd.Series)
        ), f"data must be a pandas.DataFrame, not type: {type(self.data)}"  # TODO shouldnt this only be Series?
        assert isinstance(
            self.data.index, pd.DatetimeIndex
        ), "index must be a DatetimeIndex"

        self.plot: TimeSeriesPlotter = TimeSeries.plotter(self)
        self.hist = self.plot.hist  # TODO remove this

    @property
    def time(self) -> pd.DatetimeIndex:
        return self.data.index  # type: ignore

    @property
    def start_time(self) -> datetime:
        return self.time[0]  # type: ignore

    @property
    def end_time(self) -> datetime:
        return self.time[-1]  # type: ignore

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}> '{self.name}'"

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
        self.data = self.data.loc[start_time:end_time]  # type: ignore
