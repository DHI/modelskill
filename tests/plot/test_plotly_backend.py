"""Tests that plotly is a peer backend to matplotlib.

Same plots, same arguments, and a figure returned rather than shown.
"""

import matplotlib
import numpy as np
import plotly.graph_objects as go
import pytest
from matplotlib.axes import Axes

import modelskill as ms

# every plot method that takes a `backend` argument
PLOT_KINDS = ["timeseries", "scatter", "hist", "kde", "qq", "box", "residual_hist"]
COLLECTION_PLOT_KINDS = ["scatter", "hist", "kde", "qq", "box", "residual_hist"]


@pytest.fixture(autouse=True)
def use_non_interactive_matplotlib():
    matplotlib.use("Agg")


@pytest.fixture
def o1():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return ms.PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def o2():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return ms.PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def mr1():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    return ms.model_result(fn, item=0, name="SW_1")


@pytest.fixture
def mr2():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    return ms.model_result(fn, item=0, name="SW_2")


@pytest.fixture
def cmp(o1, mr1):
    return ms.match(obs=o1, mod=mr1)


@pytest.fixture
def cmp_two_models(o1, mr1, mr2):
    return ms.match(obs=o1, mod=[mr1, mr2])


@pytest.fixture
def cc(o1, o2, mr1):
    return ms.match([o1, o2], mr1)


@pytest.fixture
def directional_cmp():
    """Comparer of a directional quantity, where axes are 0-360 degrees"""
    import pandas as pd

    time = pd.date_range("2017-01-01", periods=100, freq="h")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {"obs": rng.uniform(0, 360, 100), "model": rng.uniform(0, 360, 100)},
        index=time,
    )
    return ms.from_matched(
        df,
        obs_item="obs",
        mod_items=["model"],
        quantity=ms.Quantity("Wave direction", "degree", is_directional=True),
    )


@pytest.mark.parametrize("kind", PLOT_KINDS)
def test_comparer_plot_returns_plotly_figure(cmp, kind):
    fig = getattr(cmp.plot, kind)(backend="plotly")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


@pytest.mark.parametrize("kind", COLLECTION_PLOT_KINDS)
def test_collection_plot_returns_plotly_figure(cc, kind):
    fig = getattr(cc.plot, kind)(backend="plotly")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


@pytest.mark.parametrize("kind", PLOT_KINDS)
def test_matplotlib_is_still_the_default(cmp, kind):
    assert isinstance(getattr(cmp.plot, kind)(), Axes)


@pytest.mark.parametrize("kind", ["timeseries", "hist"])
def test_observation_plot_returns_plotly_figure(o1, kind):
    fig = getattr(o1.plot, kind)(backend="plotly")

    assert isinstance(fig, go.Figure)
    assert isinstance(getattr(o1.plot, kind)(), Axes)


@pytest.mark.parametrize("kind", PLOT_KINDS)
def test_figsize_sets_the_plotly_figure_size(cmp, kind):
    """figsize is given in inches for both backends"""
    fig = getattr(cmp.plot, kind)(backend="plotly", figsize=(4, 3))

    assert fig.layout.width == 400
    assert fig.layout.height == 300


@pytest.mark.parametrize("kind", PLOT_KINDS)
def test_title_is_set_for_both_backends(cmp, kind):
    fig = getattr(cmp.plot, kind)(backend="plotly", title="my title")
    ax = getattr(cmp.plot, kind)(title="my title")

    assert fig.layout.title.text == "my title"
    assert ax.get_title() == "my title"


@pytest.mark.parametrize("kind", PLOT_KINDS)
def test_unknown_backend_is_rejected(cmp, kind):
    with pytest.raises(ValueError, match="Invalid backend 'plotLY'"):
        getattr(cmp.plot, kind)(backend="plotLY")


@pytest.mark.parametrize("kind", PLOT_KINDS)
def test_matplotlib_axes_are_rejected_by_the_plotly_backend(cmp, kind):
    _, ax = matplotlib.pyplot.subplots()

    with pytest.raises(ValueError, match="Cannot pass matplotlib axes"):
        getattr(cmp.plot, kind)(backend="plotly", ax=ax)


@pytest.mark.parametrize("kind", PLOT_KINDS)
def test_matplotlib_only_argument_gives_an_actionable_error(cmp, kind):
    with pytest.raises(ValueError, match="Invalid plotly layout argument: 'cmap'"):
        getattr(cmp.plot, kind)(backend="plotly", cmap="OrRd")


@pytest.mark.parametrize("kind", PLOT_KINDS)
def test_plotly_layout_arguments_are_forwarded(cmp, kind):
    fig = getattr(cmp.plot, kind)(backend="plotly", width=1234)

    assert fig.layout.width == 1234


def test_plot_per_model_for_multiple_models(cmp_two_models):
    figs = cmp_two_models.plot.hist(backend="plotly")

    assert len(figs) == 2
    assert all(isinstance(f, go.Figure) for f in figs)


@pytest.mark.parametrize("kind", ["timeseries", "qq", "box"])
def test_directional_quantity_gets_a_compass_axis(directional_cmp, kind):
    fig = getattr(directional_cmp.plot, kind)(backend="plotly")

    axis = fig.layout.xaxis if kind == "qq" else fig.layout.yaxis
    assert axis.range == (0, 360)
    assert list(axis.tickvals) == list(np.linspace(0, 360, 9))


def test_scatter_skill_table_is_shown_in_plotly(cmp):
    fig = cmp.plot.scatter(backend="plotly", skill_table=True)

    assert len(fig.layout.annotations) == 1
    assert "BIAS" in fig.layout.annotations[0].text


def test_hist_density_switches_the_plotly_normalisation(cmp):
    density = cmp.plot.hist(backend="plotly", density=True)
    counts = cmp.plot.hist(backend="plotly", density=False)

    assert density.data[0].histnorm == "probability density"
    assert counts.data[0].histnorm is None
    assert density.layout.yaxis.title.text == "density"
    assert counts.layout.yaxis.title.text == "count"


def test_bin_edges_are_translated_to_plotly_bins(cmp):
    fig = cmp.plot.hist(bins=[0.0, 0.5, 1.0, 1.5], backend="plotly")

    assert fig.data[0].xbins.start == 0.0
    assert fig.data[0].xbins.end == 1.5
    assert fig.data[0].xbins.size == 0.5


def test_plotly_scatter_traces_cover_1to1_regression_points_and_quantiles(cmp):
    fig = cmp.plot.scatter(backend="plotly")

    names = [t.name for t in fig.data]
    assert "1:1" in names
    assert "Data" in names
    assert "Q-Q" in names
    assert any(n.startswith("Fit:") for n in names)
