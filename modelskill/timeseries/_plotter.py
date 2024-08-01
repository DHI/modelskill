from typing import Protocol
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class TimeSeriesPlotter(Protocol):
    def __init__(self, ts) -> None:
        pass

    def __call__(self):
        pass

    def timeseries(self):
        pass

    def hist(self):
        pass


class MatplotlibTimeSeriesPlotter(TimeSeriesPlotter):
    def __init__(self, ts) -> None:
        self._ts = ts


    def __call__(self, title=None, color=None, marker=".", linestyle="None", **kwargs):
        self.timeseries(
            title=title, color=color, marker=marker, linestyle=linestyle, **kwargs
        )

    def timeseries(
        self, title=None, color=None, marker=".", linestyle="None", **kwargs
    ):
        """Plot timeseries

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
        t = self._ts._values_as_series.index
        y = self._ts._values_as_series.values
        plt.plot(
            t,
            y,
            self._ts._values_as_series,
            marker=marker,
            linestyle=linestyle,
            color=self._ts.color if color is None else color,

        )
        ax = plt.gca()
        locator = mdates.AutoDateLocator(minticks=3, maxticks=12)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        if title:
            ax.set_title(title)
        return ax

    def hist(self, bins=100, title=None, color=None, **kwargs):
        """Plot histogram of timeseries values

        Wraps pandas.DataFrame hist() method.

        Parameters
        ----------
        bins : int, optional
            specification of bins, by default 100
        title : str, optional
            plot title, default: observation name
        color : str, optional
            plot color, by default "#d62728"
        **kwargs
            other keyword arguments to df.hist()

        Returns
        -------
        matplotlib axes
        """
        title = self._ts.name if title is None else title

        kwargs["color"] = self._ts._color if color is None else color

        ax = self._ts._values_as_series.hist(bins=bins, **kwargs)
        ax.set_title(title)
        ax.set_xlabel(str(self._ts.quantity))
        return ax


class PlotlyTimeSeriesPlotter(TimeSeriesPlotter):
    def __init__(self, ts) -> None:
        self._ts = ts

    def __call__(self):
        self.timeseries()

    def timeseries(self):
        import plotly.express as px  # type: ignore

        fig = px.line(
            self._ts._values_as_series, color_discrete_sequence=[self._ts._color]
        )
        fig.show()

    def hist(self, bins=100, **kwargs):
        """Plot histogram of timeseries values

        Wraps plotly.express.histogram() function.

        Parameters
        ----------
        bins : int, optional
            specification of bins, by default 100
        **kwargs
            other keyword arguments to df.hist()
        """
        import plotly.express as px  # type: ignore

        fig = px.histogram(
            self._ts._values_as_series,
            nbins=bins,
            color_discrete_sequence=[self._ts._color],
            **kwargs,
        )
        fig.show()
