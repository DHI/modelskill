from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

# modelskill.plotting depends on obs/model, which in turn depend on this
# module, so plotting is imported inside the methods to avoid a circular import
if TYPE_CHECKING:
    from ..plotting._backend import Backend, PlotResult


class TimeSeriesPlotter:
    """Plotter for TimeSeries (observations and model results)

    Both plots are available with the "matplotlib" (static) and the
    "plotly" (interactive) backend.

    Examples
    --------
    >>> obs.plot.timeseries()
    >>> obs.plot.hist(backend="plotly")
    """

    def __init__(self, ts) -> None:
        self._ts = ts

    def __call__(self, **kwargs) -> PlotResult:
        # default to timeseries plot
        return self.timeseries(**kwargs)

    def timeseries(
        self,
        title: str | None = None,
        color: str | None = None,
        marker: str = ".",
        linestyle: str = "None",
        ax=None,
        figsize: Tuple[float, float] | None = None,
        backend: Backend = "matplotlib",
        **kwargs: Any,
    ) -> PlotResult:
        """Plot timeseries

        Parameters
        ----------
        title : str, optional
            plot title, default: [name]
        color : str, optional
            plot color, by default '#d62728'
        marker : str, optional
            plot marker (matplotlib backend only), by default '.'
        linestyle : str, optional
            line style (matplotlib backend only), by default None
        ax : matplotlib.axes.Axes, optional
            axes to plot on (matplotlib backend only), by default None
        figsize : (float, float), optional
            figure size in inches, by default None
        backend : str, optional
            "matplotlib" (static) or "plotly" (interactive),
            by default "matplotlib"
        **kwargs
            other keyword arguments to df.plot() (matplotlib backend) or
            fig.update_layout() (plotly backend)

        Returns
        -------
        matplotlib.axes.Axes or plotly.graph_objects.Figure
        """
        from ..plotting._backend import reject_matplotlib_axes, validate_backend

        validate_backend(backend)
        reject_matplotlib_axes(ax, backend)

        ts = self._ts
        title = ts.name if title is None else title
        color = ts._color if color is None else color

        if backend == "plotly":
            from ..plotting import _plotly

            return _plotly.line(
                series=ts._values_as_series,
                color=color,
                title=title,
                ylabel=str(ts.quantity),
                figsize=figsize,
                **kwargs,
            )

        if figsize is not None:
            kwargs["figsize"] = figsize
        if ax is not None:
            kwargs["ax"] = ax

        ax = ts._values_as_series.plot(
            marker=marker, linestyle=linestyle, color=color, **kwargs
        )
        ax.set_title(title)
        ax.set_ylabel(str(ts.quantity))
        return ax

    def hist(
        self,
        bins: int = 100,
        title: str | None = None,
        color: str | None = None,
        figsize: Tuple[float, float] | None = None,
        ax=None,
        backend: Backend = "matplotlib",
        **kwargs: Any,
    ) -> PlotResult:
        """Plot histogram of timeseries values

        Parameters
        ----------
        bins : int, optional
            specification of bins, by default 100
        title : str, optional
            plot title, default: observation name
        color : str, optional
            plot color, by default "#d62728"
        figsize : (float, float), optional
            figure size in inches, by default None
        ax : matplotlib.axes.Axes, optional
            axes to plot on (matplotlib backend only), by default None
        backend : str, optional
            "matplotlib" (static) or "plotly" (interactive),
            by default "matplotlib"
        **kwargs
            other keyword arguments to df.hist() (matplotlib backend) or
            fig.update_layout() (plotly backend)

        Returns
        -------
        matplotlib.axes.Axes or plotly.graph_objects.Figure
        """
        from ..plotting._backend import reject_matplotlib_axes, validate_backend

        validate_backend(backend)
        reject_matplotlib_axes(ax, backend)

        ts = self._ts
        title = ts.name if title is None else title
        color = ts._color if color is None else color

        if backend == "plotly":
            from ..plotting import _plotly

            return _plotly.histogram(
                series={ts.name: ts._values_as_series.values},
                colors=[color],
                bins=bins,
                density=False,
                alpha=1.0,
                title=title,
                xlabel=str(ts.quantity),
                figsize=figsize,
                **kwargs,
            )

        if figsize is not None:
            kwargs["figsize"] = figsize
        if ax is not None:
            kwargs["ax"] = ax

        ax = ts._values_as_series.hist(bins=bins, color=color, **kwargs)
        ax.set_title(title)
        ax.set_xlabel(str(ts.quantity))
        return ax


# kept as an alias: the plotter used to be selected by class, it is now
# selected by the `backend` argument on each plot method
MatplotlibTimeSeriesPlotter = TimeSeriesPlotter
