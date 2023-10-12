import pytest
import xarray as xr
import numpy as np
import pandas as pd
from modelskill.timeseries import TimeSeries
from modelskill.timeseries._plotter import MatplotlibTimeSeriesPlotter
import matplotlib.pyplot as plt


@pytest.fixture()
def timeseries():
    times = pd.date_range("2000-01-01", periods=100)
    data1 = np.arange(100)
    ds = xr.Dataset(
        {
            "modelskill-rocks": ("time", data1, {"kind": "observation"}),
            "x": None,
            "y": None,
        },
        coords={"time": times},
        attrs={
            "gtype": "point",
        },
    )

    ts = TimeSeries(ds)
    return ts


@pytest.fixture(params=["plot", "hist", "timeseries"])
def matplotlib_timeseries_plotting_function(request, timeseries):
    plotter = MatplotlibTimeSeriesPlotter(timeseries)
    return getattr(plotter, request.param)


def test_matplotlib_timeseries_basic_plots_work(
    matplotlib_timeseries_plotting_function,
):
    matplotlib_timeseries_plotting_function()
    assert True


def test_matplotlib_timeseries_accept_ax(
    matplotlib_timeseries_plotting_function,
):
    _, ax = plt.subplots()
    ret_ax = matplotlib_timeseries_plotting_function(ax=ax)
    assert ret_ax is not None, "Return object should not be None"
    assert ret_ax is ax


def test_matplotlib_timeseries_accepts_figsize(
    matplotlib_timeseries_plotting_function,
):
    figsize = (20, 10)
    ax = matplotlib_timeseries_plotting_function(figsize=figsize)
    a, b = ax.get_figure().get_size_inches()
    assert a, b == figsize


def test_matplotlib_timeseries_accepts_title(
    matplotlib_timeseries_plotting_function,
):
    title = "Modelskill is fun!"
    ax = matplotlib_timeseries_plotting_function(title=title)
    assert ax.get_title() == title
