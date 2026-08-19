"""Tests that plotly is a peer backend to matplotlib.

Same plots, same arguments, and a figure returned rather than shown.
"""

import sys

import matplotlib
import matplotlib.figure
import numpy as np
import plotly.graph_objects as go
import pytest
from matplotlib.axes import Axes

import modelskill as ms
from modelskill.plotting import _plotly

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


def test_directional_scatter_gets_a_compass_axis_on_both_backends(directional_cmp):
    compass = list(np.linspace(0, 360, 9))
    fig = directional_cmp.plot.scatter(backend="plotly")
    ax = directional_cmp.plot.scatter()

    assert list(fig.layout.xaxis.tickvals) == compass
    assert list(fig.layout.yaxis.tickvals) == compass
    assert fig.layout.xaxis.range == (0, 360)
    assert fig.layout.yaxis.range == (0, 360)
    assert list(ax.get_xticks()) == compass
    assert list(ax.get_yticks()) == compass
    assert ax.get_xlim() == (0.0, 360.0)
    assert ax.get_ylim() == (0.0, 360.0)


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


def test_non_uniform_bin_edges_are_rejected_by_the_plotly_backend(cmp):
    with pytest.raises(ValueError, match="uniformly spaced bin edges"):
        cmp.plot.hist(bins=[0.0, 0.5, 1.0, 3.0], backend="plotly")


def test_plotly_scatter_traces_cover_1to1_regression_points_and_quantiles(cmp):
    fig = cmp.plot.scatter(backend="plotly")

    names = [t.name for t in fig.data]
    assert "1:1" in names
    assert "Data" in names
    assert "Q-Q" in names
    assert any(n.startswith("Fit:") for n in names)


@pytest.mark.parametrize("normalize_std", [False, True])
def test_taylor_returns_a_polar_plotly_figure(cc, normalize_std):
    fig = cc.plot.taylor(backend="plotly", normalize_std=normalize_std)

    assert isinstance(fig, go.Figure)
    assert fig.layout.polar.sector == (0, 90)
    assert all(t.type == "scatterpolar" for t in fig.data)


def test_taylor_places_the_models_at_arccos_of_the_correlation(cmp):
    from modelskill import metrics as mtr

    sk = cmp.skill(metrics=[mtr.cc, mtr._std_mod]).to_dataframe()
    fig = cmp.plot.taylor(backend="plotly")

    # a single-model Comparer labels its taylor point "model", not the model name
    model = fig.data[-1]
    assert model.theta[0] == pytest.approx(
        np.degrees(np.arccos(sk["cc"].iloc[0])), abs=1e-6
    )
    assert model.r[0] == pytest.approx(sk["_std_mod"].iloc[0], rel=1e-6)


def test_taylor_matplotlib_still_returns_a_matplotlib_figure(cc):
    assert isinstance(cc.plot.taylor(), matplotlib.figure.Figure)


def test_temporal_coverage_has_one_row_per_data_source(o1, o2, mr1):
    fig = ms.plotting.temporal_coverage([o1, o2], mr1, backend="plotly")

    assert isinstance(fig, go.Figure)
    assert [t.name for t in fig.data] == ["SW_1", "HKNA", "EPL"]


def test_temporal_coverage_limits_the_time_axis_to_the_model_period(o1, mr1):
    limited = ms.plotting.temporal_coverage(o1, mr1, backend="plotly")
    unlimited = ms.plotting.temporal_coverage(
        o1, mr1, limit_to_model_period=False, backend="plotly"
    )

    assert limited.layout.xaxis.range is not None
    assert unlimited.layout.xaxis.range is None


def test_spatial_overview_shows_the_domain_and_the_observations(o1, o2, mr1):
    fig = ms.plotting.spatial_overview([o1, o2], mr1, backend="plotly")

    assert isinstance(fig, go.Figure)
    assert [t.name for t in fig.data] == ["Domain", "HKNA", "EPL"]
    # equal aspect ratio, as for a map
    assert fig.layout.yaxis.scaleanchor == "x"


def test_spatial_overview_track_observations_are_drawn_as_points(o1, mr1):
    track = ms.TrackObservation(
        "tests/testdata/SW/Alti_c2_Dutch.dfs0", item=3, name="c2"
    )
    fig = ms.plotting.spatial_overview([o1, track], mr1, backend="plotly")

    c2 = [t for t in fig.data if t.name == "c2"][0]
    assert c2.mode == "markers"
    assert len(c2.x) == track.n_points


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_spatial_overview_rejects_an_unsupported_observation(o1, mr1, backend):
    class NotAnObservation:
        name = "nope"

    with pytest.raises(ValueError, match="Could not show observation"):
        ms.plotting.spatial_overview([o1, NotAnObservation()], mr1, backend=backend)


@pytest.fixture
def wave_dir_dataframe():
    import mikeio

    ds = mikeio.read("tests/testdata/wave_dir.dfs0")
    return ds[[0, 2, 1, 3]].to_dataframe()


def test_wind_rose_is_a_stacked_polar_bar_chart(wave_dir_dataframe):
    fig = ms.plotting.wind_rose(wave_dir_dataframe, backend="plotly")

    assert isinstance(fig, go.Figure)
    assert fig.layout.barmode == "stack"
    assert all(t.type == "barpolar" for t in fig.data)
    # north up, clockwise, as in the matplotlib version
    assert fig.layout.polar.angularaxis.direction == "clockwise"
    assert fig.layout.polar.angularaxis.rotation == 90


def test_wind_rose_dual_has_a_legend_group_per_dataset(wave_dir_dataframe):
    dual = ms.plotting.wind_rose(wave_dir_dataframe, backend="plotly")
    single = ms.plotting.wind_rose(wave_dir_dataframe.iloc[:, :2], backend="plotly")

    assert set(t.legendgroup for t in dual.data) == {"Measurement", "Model"}
    assert set(t.legendgroup for t in single.data) == {"Measurement"}
    assert len(dual.data) == 2 * len(single.data)


def test_wind_rose_has_compass_labels_like_matplotlib(wave_dir_dataframe):
    fig = ms.plotting.wind_rose(wave_dir_dataframe, backend="plotly")

    labels = list(fig.layout.polar.angularaxis.ticktext)
    assert labels[:5] == ["N", "NNE", "NE", "ENE", "E"]
    assert list(fig.layout.polar.angularaxis.tickvals)[:3] == [0.0, 22.5, 45.0]


def test_wind_rose_calm_becomes_the_polar_hole(wave_dir_dataframe):
    fig = ms.plotting.wind_rose(wave_dir_dataframe, backend="plotly")

    assert 0 < fig.layout.polar.hole < 1


def test_wind_rose_matplotlib_is_unchanged(wave_dir_dataframe):
    ax = ms.plotting.wind_rose(wave_dir_dataframe)

    assert ax.name == "polar"


# --- plotly layout interop ---


def test_import_plotly_go_missing_dependency_gives_actionable_error(monkeypatch):
    # setting a sys.modules entry to None makes the import raise ImportError
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)

    with pytest.raises(ImportError, match=r'pip install "modelskill\[plotly\]"'):
        _plotly.import_plotly_go()


def test_figsize_is_translated_to_plotly_pixels():
    assert _plotly.figsize_to_layout(None) == {}
    assert _plotly.figsize_to_layout((8, 6)) == {"width": 800, "height": 600}


def test_apply_layout_uses_figsize_for_width_and_height():
    fig = _plotly.apply_layout(go.Figure(), figsize=(3, 4))

    assert fig.layout.width == 300
    assert fig.layout.height == 400


def test_apply_layout_lets_explicit_width_win_over_figsize():
    fig = _plotly.apply_layout(go.Figure(), figsize=(3, 4), width=1000)

    assert fig.layout.width == 1000
    assert fig.layout.height == 400


def test_apply_layout_ignores_none_values():
    fig = _plotly.apply_layout(go.Figure(), figsize=None, title=None)

    assert fig.layout.width is None
    assert fig.layout.title.text is None


def test_apply_layout_names_the_offending_matplotlib_argument():
    with pytest.raises(ValueError, match="Invalid plotly layout argument: 'cmap'"):
        _plotly.apply_layout(go.Figure(), cmap="OrRd")
