import numpy as np
import pytest
import pandas as pd
import xarray as xr
import modelskill.comparison
import modelskill as ms

import matplotlib as mpl
import matplotlib.pyplot as plt

# use non-interactive backend for testing
mpl.use("Agg")


def _set_attrs(data: xr.Dataset) -> xr.Dataset:
    data["Observation"].attrs["kind"] = "observation"
    data["Observation"].attrs["long_name"] = "fake var"
    data["Observation"].attrs["units"] = "m"
    data["m1"].attrs["kind"] = "model"
    data["m2"].attrs["kind"] = "model"
    data.attrs["modelskill_version"] = modelskill.__version__
    return data


@pytest.fixture
def pc() -> modelskill.comparison.Comparer:
    """A comparer with fake point data and 2 models"""
    x, y = 10.0, 55.0
    df = pd.DataFrame(
        {
            "Observation": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
            "m1": [1.5, 2.4, 3.6, 4.9, 5.6, 6.4],
            "m2": [1.1, 2.2, 3.1, 4.2, 4.9, 6.2],
        },
        index=pd.date_range("2019-01-01", periods=6, freq="D"),
    )
    df.index.name = "time"

    data = df.to_xarray()
    data.attrs["gtype"] = "point"
    data.attrs["name"] = "fake point obs"
    data.coords["x"] = x
    data.coords["y"] = y
    data.coords["z"] = np.nan
    data = _set_attrs(data)

    raw_data = {"m1": data[["m1"]].copy(), "m2": data[["m2"]].copy()}

    data = data.dropna(dim="time")
    return modelskill.comparison.Comparer(matched_data=data, raw_mod_data=raw_data)


@pytest.fixture
def tc() -> modelskill.comparison.Comparer:
    """A comparer with fake track data and 3 models"""
    df = pd.DataFrame(
        {
            "Observation": [-1.0, -2.0, -3.0, -4.0, -5.0, np.nan],
            "x": [10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
            "y": [55.1, 55.2, 55.3, 55.4, 55.5, 55.6],
            "m1": [-1.5, -2.4, -3.6, -4.9, -5.6, -6.4],
            "m2": [-1.1, -2.2, -3.1, -4.2, -4.9, -6.2],
            "m3": [-1.3, -2.3, -3.3, -4.3, -5.3, -6.3],
        },
        index=pd.date_range("2019-01-03", periods=6, freq="D"),
    )
    df.index.name = "time"
    data = df.to_xarray()
    data = data.set_coords(["x", "y"])
    data.attrs["gtype"] = "track"
    data.attrs["name"] = "fake track obs"
    data["m3"].attrs["kind"] = "model"
    data = _set_attrs(data)

    raw_data = {
        "m1": data[["m1"]].copy(),
        "m2": data[["m2"]].copy(),
        "m3": data[["m3"]].copy(),
    }

    data = data.dropna(dim="time")
    return modelskill.comparison.Comparer(matched_data=data, raw_mod_data=raw_data)


@pytest.fixture
def cc(pc, tc) -> modelskill.ComparerCollection:
    """A comparer collection with two comparers, with partial overlap in time
    one comparer with 2 models, one comparer with 3 models"""
    return modelskill.ComparerCollection([pc, tc])


def test_cc_properties(cc):
    assert cc.n_comparers == 2
    assert len(cc) == 2
    assert cc.n_models == 3  # first:2, second:3
    assert cc.n_points == 10  # 5 + 5
    assert cc.start_time == pd.Timestamp("2019-01-01")
    assert cc.end_time == pd.Timestamp("2019-01-07")
    assert cc.obs_names == ["fake point obs", "fake track obs"]
    assert cc.mod_names == ["m1", "m2", "m3"]


def test_cc_sel_model(cc):
    cc2 = cc.sel(model="m1")
    assert cc2.n_comparers == 2
    assert cc2.n_models == 1
    assert cc2.n_points == 10
    assert cc2.start_time == pd.Timestamp("2019-01-01")
    assert cc2.end_time == pd.Timestamp("2019-01-07")
    assert cc2.obs_names == ["fake point obs", "fake track obs"]
    assert cc2.mod_names == ["m1"]


def test_cc_sel_model_m3(cc):
    cc2 = cc.sel(model="m3")
    assert cc2.n_comparers == 1
    assert cc2.n_models == 1


def test_cc_sel_model_last(cc):
    # last is m3 which is not in the first comparer
    cc2 = cc.sel(model=-1)
    assert cc2.n_comparers == 1
    assert cc2.n_models == 1
    assert cc2.n_points == 5
    assert cc2.start_time == pd.Timestamp("2019-01-03")
    assert cc2.end_time == pd.Timestamp("2019-01-07")
    assert cc2.obs_names == ["fake track obs"]
    assert cc2.mod_names == ["m3"]


# TODO: FAILS
# def test_cc_sel_time_single(cc):
#     cc1 = cc.sel(time="2019-01-03")
#     assert cc1.n_comparers == 2
#     assert cc1.n_models == 3
#     assert cc1.n_points == 6
#     assert cc1.start == pd.Timestamp("2019-01-03")
#     assert cc1.end == pd.Timestamp("2019-01-05")
#     assert cc1.obs_names == ["fake point obs", "fake track obs"]
#     assert cc1.mod_names == ["m1", "m2", "m3"]


def test_cc_sel_time(cc):
    cc2 = cc.sel(time=slice("2019-01-03", "2019-01-05"))
    assert cc2.n_comparers == 2
    assert cc2.n_models == 3
    assert cc2.n_points == 6
    assert cc2.start_time == pd.Timestamp("2019-01-03")
    assert cc2.end_time == pd.Timestamp("2019-01-05")
    assert cc2.obs_names == ["fake point obs", "fake track obs"]
    assert cc2.mod_names == ["m1", "m2", "m3"]


def test_cc_sel_attrs(cc):
    cc2 = cc.sel(gtype="point")
    assert cc2.n_comparers == 1
    assert cc2[0].gtype == "point"


def test_cc_query(cc):
    cc2 = cc.query("Observation > 3")
    assert cc2.n_comparers == 1
    assert cc2.n_models == 2
    assert cc2.n_points == 2


def test_save_and_load_preserves_order_of_comparers(tmp_path):
    data = pd.DataFrame(
        {"zulu": [1, 2, 3], "alpha": [4, 5, 6], "bravo": [7, 8, 9], "m1": [10, 11, 12]}
    )

    cmp1 = ms.from_matched(data, obs_item="zulu", mod_items="m1")
    cmp2 = ms.from_matched(data, obs_item="alpha", mod_items="m1")
    cmp3 = ms.from_matched(data, obs_item="bravo", mod_items="m1")

    cc = ms.ComparerCollection([cmp1, cmp2, cmp3])
    assert cc[0].name == "zulu"
    assert cc[1].name == "alpha"
    assert cc[2].name == "bravo"

    fn = tmp_path / "test_cc.msk"
    cc.save(fn)

    cc2 = modelskill.load(fn)
    assert cc2[0].name == "zulu"
    assert cc2[1].name == "alpha"
    assert cc2[2].name == "bravo"


def test_save(cc: modelskill.ComparerCollection, tmp_path):
    fn = tmp_path / "test_cc.msk"
    assert cc[0].data.attrs["modelskill_version"] == modelskill.__version__
    cc.save(fn)

    cc2 = modelskill.load(fn)
    assert cc2.n_comparers == 2

    # this belongs to the comparer, but ComparerCollection is the commonly used class
    assert cc[0].data.attrs["modelskill_version"] == modelskill.__version__


def test_load_from_root_module(cc, tmp_path):
    fn = tmp_path / "test_cc.msk"
    cc.save(fn)

    cc2 = modelskill.load(fn)
    assert cc2.n_comparers == 2


def test_save_and_load_preserves_raw_model_data(cc, tmp_path):
    fn = tmp_path / "test_cc.msk"
    assert len(cc["fake point obs"].raw_mod_data["m1"]) == 6
    cc.save(fn)

    cc2 = modelskill.load(fn)

    # we ideally would like to test is the original raw_mod_data is fully included in this plot
    cc2[0].plot.timeseries()

    # for now, we just test if the raw_mod_data is full length
    assert len(cc2["fake point obs"].raw_mod_data["m1"]) == 6


def test_scatter(cc):
    ax = cc.plot.scatter(skill_table=True)
    assert ax is not None


def test_hist(cc):
    ax = cc.sel(model="m1").plot.hist()
    assert ax is not None


def test_kde(cc):
    ax = cc.plot.kde()
    assert ax is not None


def test_add_cc_pc(cc, pc):
    pc2 = pc.copy()
    pc2.data.attrs["name"] = "pc2"
    cc2 = cc + pc2
    assert cc2.n_points == 15
    assert cc2.n_comparers == 3


def test_add_cc_tc(cc, tc):
    tc2 = tc.copy()
    tc2.data.attrs["name"] = "tc2"
    cc2 = cc + tc2
    assert cc2.n_points == 15
    assert cc2.n_comparers == 3


def test_add_cc_cc(cc, pc, tc):
    pc2 = pc.copy()
    pc2.data.attrs["name"] = "pc2"
    tc2 = tc.copy()
    tc2.data.attrs["name"] = "tc2"
    tc3 = tc.copy()  # keep name
    cc2 = pc2 + tc2 + tc3

    cc3 = cc + cc2
    # assert cc3.n_points == 15
    assert cc3.n_comparers == 4


def test_plots_directional(cc):
    cc = cc.sel(model="m1")

    cc.plot.is_directional = True

    ax = cc.plot.scatter()
    assert "m1" in ax.get_title()
    assert ax.get_xlim() == (0.0, 360.0)
    assert ax.get_ylim() == (0.0, 360.0)
    assert len(ax.get_legend().get_texts()) == 1  # no reg line or qq

    ax = cc.plot.kde()
    assert ax is not None
    assert ax.get_xlim() == (0.0, 360.0)

    ax = cc.plot.hist()
    assert ax is not None
    assert ax.get_xlim() == (0.0, 360.0)


@pytest.fixture(
    params=[
        "scatter",
        "kde",
        "hist",
        "taylor",
    ]
)
def cc_plot_function(cc, request):
    # TODO - fix, then remove this block
    if request.param == "taylor":
        pytest.skip(
            "taylor plot fails due to mean_skill() on collections, needs investigation"
        )

    func = getattr(cc.plot, request.param)
    # special cases require selecting a model
    if request.param in ["scatter", "hist"]:

        def func(**kwargs):
            kwargs["model"] = 0
            wrapped_func = getattr(cc.plot, request.param)
            return wrapped_func(**kwargs)

    return func


def test_plot_returns_an_object(cc_plot_function):
    obj = cc_plot_function()
    assert obj is not None


def test_plot_accepts_ax_if_relevant(cc_plot_function):
    _, ax = plt.subplots()
    func_name = cc_plot_function.__name__
    # plots that don't accept ax
    if func_name in ["taylor"]:
        return
    ret_ax = cc_plot_function(ax=ax)
    assert ret_ax is ax


def test_plot_accepts_title(cc_plot_function):
    expected_title = "test title"
    ret_obj = cc_plot_function(title=expected_title)

    # Handle both ax and fig titles
    title = None
    if hasattr(ret_obj, "get_title"):
        title = ret_obj.get_title()
    elif hasattr(ret_obj, "get_suptitle"):
        title = ret_obj.get_suptitle()
    elif hasattr(ret_obj, "_suptitle"):  # older versions of matplotlib
        title = ret_obj._suptitle.get_text()
    else:
        raise pytest.fail("Could not access title from return object.")

    assert title == expected_title


def test_plot_accepts_figsize(cc_plot_function):
    figsize = (10, 10)
    ax = cc_plot_function(figsize=figsize)
    a, b = ax.get_figure().get_size_inches()
    assert a, b == figsize


def test_filter_by_attrs(cc):
    cc2 = cc.filter_by_attrs(gtype="point")
    assert cc2.n_comparers == 1
    assert cc2[0].gtype == "point"


def test_filter_by_attrs_custom(cc):
    cc[0].data.attrs["custom"] = 12
    cc[1].data.attrs["custom"] = 13

    cc2 = cc.filter_by_attrs(custom=12)
    assert cc2.n_comparers == 1
    assert cc2[0].data.attrs["custom"] == 12
    assert cc2[0] == cc[0]

    cc[0].data.attrs["custom2"] = True
    cc3 = cc.filter_by_attrs(custom2=True)
    assert cc3.n_comparers == 1
    assert cc3[0].data.attrs["custom2"]
    assert cc3[0] == cc[0]
