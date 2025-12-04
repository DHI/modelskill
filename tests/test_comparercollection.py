import numpy as np
import pytest
import pandas as pd
import xarray as xr
import modelskill.comparison
import modelskill as ms

import matplotlib as mpl
import matplotlib.pyplot as plt

from modelskill.model.point import PointModelResult

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
def cc_pr() -> modelskill.comparison.Comparer:
    """Real data to test Peak Ratio from top-to-bottom"""
    obs = modelskill.PointObservation(
        "tests/testdata/PR_test_data.dfs0", item="Hs_measured"
    )
    mod = modelskill.PointModelResult(
        "tests/testdata/PR_test_data.dfs0", item="Hs_model"
    )
    return modelskill.match(obs, mod)


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

    raw_data = {
        "m1": PointModelResult(data[["m1"]].copy()),
        "m2": PointModelResult(data[["m2"]].copy()),
    }

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

    data = data.dropna(dim="time")
    return modelskill.comparison.Comparer(matched_data=data)


@pytest.fixture
def cc(pc, tc) -> ms.ComparerCollection:
    """A comparer collection with two comparers, with partial overlap in time
    one comparer with 2 models, one comparer with 3 models"""
    return ms.ComparerCollection([pc, tc])


def test_cc_properties(cc):
    assert len(cc) == 2
    assert len(cc) == 2
    assert cc.n_models == 3  # first:2, second:3
    assert cc.n_points == 10  # 5 + 5
    assert cc.start_time == pd.Timestamp("2019-01-01")
    assert cc.end_time == pd.Timestamp("2019-01-07")
    assert cc.obs_names == ["fake point obs", "fake track obs"]
    assert cc.mod_names == ["m1", "m2", "m3"]


def test_cc_sel_model(cc):
    cc2 = cc.sel(model="m1")
    assert len(cc2) == 2
    assert cc2.n_models == 1
    assert cc2.n_points == 10
    assert cc2.start_time == pd.Timestamp("2019-01-01")
    assert cc2.end_time == pd.Timestamp("2019-01-07")
    assert cc2.obs_names == ["fake point obs", "fake track obs"]
    assert cc2.mod_names == ["m1"]


def test_cc_sel_model_m3(cc):
    cc2 = cc.sel(model="m3")
    assert len(cc2) == 1
    assert cc2.n_models == 1


def test_cc_sel_model_last(cc):
    # last is m3 which is not in the first comparer
    cc2 = cc.sel(model=-1)
    assert len(cc2) == 1
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
    assert len(cc2) == 2
    assert cc2.n_models == 3
    assert cc2.n_points == 6
    assert cc2.start_time == pd.Timestamp("2019-01-03")
    assert cc2.end_time == pd.Timestamp("2019-01-05")
    assert cc2.obs_names == ["fake point obs", "fake track obs"]
    assert cc2.mod_names == ["m1", "m2", "m3"]


def test_cc_sel_attrs(cc):
    cc2 = cc.sel(gtype="point")
    assert len(cc2) == 1
    assert cc2[0].gtype == "point"


def test_cc_query(cc):
    cc2 = cc.query("Observation > 3")
    assert len(cc2) == 1
    assert cc2.n_models == 2
    assert cc2.n_points == 2


def test_merge_cc_pc(cc, pc):
    pc2 = pc.copy()
    pc2.data.attrs["name"] = "pc2"
    cc2 = cc.merge(pc2)
    assert cc2.n_points == 15
    assert len(cc2) == 3


def test_merge_cc_tc(cc, tc):
    tc2 = tc.copy()
    tc2.data.attrs["name"] = "tc2"
    cc2 = cc.merge(tc2)
    assert cc2.n_points == 15
    assert len(cc2) == 3


def test_add_cc_cc(cc, pc, tc):
    pc2 = pc.copy()
    pc2.data.attrs["name"] = "pc2"
    tc2 = tc.copy()
    tc2.data.attrs["name"] = "tc2"
    tc3 = tc.copy()  # keep name
    cc2 = pc2.merge(tc2).merge(tc3)

    cc3 = cc.merge(cc2)
    # assert cc3.n_points == 15
    assert len(cc3) == 4


def test_rename_obs(cc):
    cc2 = cc.rename({"fake point obs": "fake point obs 2"})
    assert cc2.obs_names == ["fake point obs 2", "fake track obs"]
    assert cc.obs_names == ["fake point obs", "fake track obs"]

    cc3 = cc.rename(
        {"fake point obs": "fake point obs 2", "fake track obs": "fake track obs 2"}
    )
    assert cc3.obs_names == ["fake point obs 2", "fake track obs 2"]


def test_rename_mod(cc):
    cc2 = cc.rename({"m1": "m1b"})
    assert cc2.mod_names == ["m1b", "m2", "m3"]
    assert cc.mod_names == ["m1", "m2", "m3"]

    cc3 = cc.rename({"m1": "m1b", "m2": "m2b", "m3": "m3b"})
    assert cc3.mod_names == ["m1b", "m2b", "m3b"]


def test_rename_mod_and_obs(cc):
    cc2 = cc.rename({"m1": "m1b", "fake point obs": "fake point obs 2"})
    assert cc2.mod_names == ["m1b", "m2", "m3"]
    assert cc2.obs_names == ["fake point obs 2", "fake track obs"]
    assert cc.mod_names == ["m1", "m2", "m3"]
    assert cc.obs_names == ["fake point obs", "fake track obs"]


def test_rename_aux(cc):
    aux = xr.ones_like(cc[0].data["m1"])
    aux.attrs["kind"] = "aux"
    cc[0].data["aux"] = aux
    assert "aux" in cc.aux_names
    cc2 = cc.rename({"aux": "aux2"})
    assert "aux" not in cc2[0].data
    assert cc2.aux_names == ["aux2"]


def test_rename_aux_and_mod(cc):
    aux = xr.ones_like(cc[0].data["m1"])
    aux.attrs["kind"] = "aux"
    cc[0].data["aux"] = aux
    cc2 = cc.rename({"aux": "aux2", "m1": "m1b"})
    assert cc2.aux_names == ["aux2"]
    assert cc2.mod_names == ["m1b", "m2", "m3"]


def test_rename_fails_key_error(cc):
    with pytest.raises(KeyError):
        cc.rename({"m1": "m1b", "fake point obs": "fake point obs 2", "m4": "m4b"})
    with pytest.raises(KeyError):
        cc.rename({"m4": "m4b"})
    with pytest.raises(KeyError):
        cc.rename(
            {
                "fake point obs": "fake point obs 2",
                "fake track obs": "fake track obs 2",
                "m4": "m4b",
            }
        )


def test_rename_fails_reserved_names(cc):
    with pytest.raises(ValueError, match="reserved names!"):
        cc.rename({"m1": "x"})
    with pytest.raises(ValueError, match="reserved names!"):
        cc.rename({"m1": "MOD1", "m2": "y"})
    with pytest.raises(ValueError, match="reserved names!"):
        cc.rename({"m1": "z", "fake point obs": "OBS"})
    with pytest.raises(ValueError, match="reserved names!"):
        cc.rename({"m1": "time"})
    with pytest.raises(ValueError, match="reserved names!"):
        cc.rename({"m1": "Observation"})


def test_filter_by_attrs(cc):
    cc2 = cc.filter_by_attrs(gtype="point")
    assert len(cc2) == 1
    assert cc2[0].gtype == "point"


def test_filter_by_attrs_custom(cc):
    cc[0].data.attrs["custom"] = 12
    cc[1].data.attrs["custom"] = 13

    cc2 = cc.filter_by_attrs(custom=12)
    assert len(cc2) == 1
    assert cc2[0].data.attrs["custom"] == 12

    cc[0].data.attrs["custom2"] = True
    cc3 = cc.filter_by_attrs(custom2=True)
    assert len(cc3) == 1
    assert cc3[0].data.attrs["custom2"]


def test_skill_by_attrs_gtype(cc):
    sk = cc.skill(by="attrs:gtype")
    assert len(sk) == 2
    assert sk.data.index[0] == "point"
    assert sk.data.index[1] == "track"
    assert sk.data.index.name == "gtype"


def test_skill_by_freq(cc):
    skd = cc.skill(by="freq:D")
    assert len(skd) == 7

    skw = cc.skill(by="freq:W")
    assert len(skw) == 2


def test_skill_by_attrs_gtype_and_mod(cc):
    sk = cc.skill(by=["attrs:gtype", "model"])
    assert len(sk) == 5
    assert sk.data.index[0] == ("point", "m1")
    assert sk.data.index[1] == ("point", "m2")
    assert sk.data.index[2] == ("track", "m1")
    assert sk.data.index[3] == ("track", "m2")
    assert sk.data.index[4] == ("track", "m3")
    assert sk.data.index.names[0] == "gtype"
    assert sk.data.index.names[1] == "model"

    # TODO: observed=True doesn't work on model
    # sk2 = cc.skill(by=["attrs:gtype", "model"], observed=True)
    # assert len(sk2) == 6
    # assert sk.data.index[0] == ("point", "m1")
    # assert sk.data.index[1] == ("point", "m2")
    # assert sk.data.index[2] == ("point", "m3")


def test_skill_by_attrs_int(cc):
    cc[0].data.attrs["custom"] = 12
    cc[1].data.attrs["custom"] = 13

    sk = cc.skill(by="attrs:custom")
    assert len(sk) == 2
    assert sk.data.index[0] == 12
    assert sk.data.index[1] == 13
    assert sk.data.index.name == "custom"

    sk = cc.skill(by=("attrs:custom", "model"))
    assert len(sk) == 5
    assert sk.data.index[4] == (13, "m3")


def test_skill_by_attrs_observed(cc):
    cc[0].data.attrs["use"] = "DA"  # point

    sk = cc.skill(by="attrs:use")  # observed=False is default
    assert len(sk) == 2
    assert sk.data.index[0] == "DA"
    assert sk.data.index[1] is False
    assert sk.data.index.name == "use"

    sk = cc.skill(by="attrs:use", observed=True)
    assert len(sk) == 1
    assert sk.data.index[0] == "DA"
    assert sk.data.index.name == "use"


def test_xy_in_skill(cc):
    # point obs has x,y, track obs x, y are np.nan
    sk = cc.skill()
    assert "x" in sk.data.columns
    assert "y" in sk.data.columns
    df = sk.data.reset_index()
    df_track = df.loc[df.observation == "fake track obs"]
    assert df_track.x.isna().all()
    assert df_track.y.isna().all()
    df_point = df.loc[df.observation == "fake point obs"]
    assert all(df_point.x == cc[0].x)
    assert all(df_point.y == cc[0].y)


def test_xy_in_skill_no_obs(cc):
    # if no observation column then no x, y information!
    # e.g. if we filter by gtype (in this case 1 per obs), no x, y information
    sk = cc.skill(by=["attrs:gtype", "model"])
    assert "x" in sk.data.columns
    assert "y" in sk.data.columns
    df = sk.data.reset_index()
    assert df.x.isna().all()
    assert df.y.isna().all()


# ======================== load/save ========================


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
    assert len(cc2) == 2

    # this belongs to the comparer, but ComparerCollection is the commonly used class
    assert cc[0].data.attrs["modelskill_version"] == modelskill.__version__


def test_load_from_root_module(cc, tmp_path):
    fn = tmp_path / "test_cc.msk"
    cc.save(fn)

    cc2 = modelskill.load(fn)
    assert len(cc2) == 2


def test_save_and_load_preserves_raw_model_data(cc, tmp_path):
    fn = tmp_path / "test_cc.msk"
    assert len(cc["fake point obs"].raw_mod_data["m1"]) == 6
    cc.save(fn)

    cc2 = modelskill.load(fn)

    # we ideally would like to test is the original raw_mod_data is fully included in this plot
    cc2[0].plot.timeseries()

    # for now, we just test if the raw_mod_data is full length
    assert len(cc2["fake point obs"].raw_mod_data["m1"]) == 6
    assert cc2["fake point obs"].raw_mod_data["m1"].name == "m1"


# ======================== plotting ========================


def test_plot_scatter(cc):
    ax = cc.plot.scatter(skill_table=True)
    assert ax is not None


def test_plot_hist(cc):
    ax = cc.sel(model="m1").plot.hist()
    assert ax is not None


def test_plot_kde(cc):
    ax = cc.plot.kde()
    assert ax is not None


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


PLOT_FUNCS_RETURNING_MANY_AX = ["scatter", "hist", "residual_hist"]


@pytest.fixture(
    params=[
        "scatter",
        "kde",
        "hist",
        "taylor",
        "box",
        "qq",
        "residual_hist",
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
    if request.param in PLOT_FUNCS_RETURNING_MANY_AX:

        def func(**kwargs):
            wrapped_func = getattr(cc.sel(model=[0]).plot, request.param)
            return wrapped_func(**kwargs)

    return func


@pytest.mark.parametrize("kind", PLOT_FUNCS_RETURNING_MANY_AX)
def test_plots_returning_multiple_axes(pc, kind):
    n_models = 2
    func = getattr(pc.plot, kind)
    ax = func()
    assert len(ax) == n_models
    assert all(isinstance(a, plt.Axes) for a in ax)


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


def test_peak_ratio(cc):
    """Non existent peak ratio"""
    cc = cc.sel(model="m1")
    sk = cc.skill(metrics=["peak_ratio"])

    assert sk.loc["fake point obs", "peak_ratio"] == pytest.approx(1.119999999)


def test_peak_ratio_2(cc_pr):
    sk = cc_pr.skill(metrics=["peak_ratio"])
    assert "peak_ratio" in sk.data.columns
    assert sk.to_dataframe()["peak_ratio"].values == pytest.approx(0.88999995)


def test_copy(cc):
    cc2 = cc.copy()
    assert cc2.n_models == 3
    assert cc2.n_points == 10
    assert cc2.start_time == pd.Timestamp("2019-01-01")
    assert cc2.end_time == pd.Timestamp("2019-01-07")
    assert cc2.obs_names == ["fake point obs", "fake track obs"]
    assert cc2.mod_names == ["m1", "m2", "m3"]


def test_plot_spatial_overview(cc):
    ax = cc.plot.spatial_overview()
    # TODO add sensible assertions
    assert ax is not None


def test_plot_temporal_coverage(cc):
    ax = cc.plot.temporal_coverage()
    # TODO add more sensible assertions
    lines = ax.get_lines()
    assert len(lines) == 4  # 1 point, 1 track, 2 models
    assert ax is not None


def test_handle_no_overlap_in_time():
    o1 = ms.PointObservation(
        pd.DataFrame({"O1": np.zeros(2)}, index=pd.date_range("2000", periods=2)),
        # inside the domain
        x=0.25,
        y=0.25,
    )
    o2 = ms.PointObservation(
        # mismatch in time
        pd.DataFrame({"O2": np.zeros(2)}, index=pd.date_range("2100", periods=2)),
        # inside the domain
        x=0.5,
        y=0.5,
    )

    mod = ms.GridModelResult(
        xr.DataArray(
            name="foo",
            data=np.zeros((2, 2, 2)),
            dims=["time", "x", "y"],
            coords={
                "time": pd.date_range("2000", periods=2),
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
            },
        )
    )

    obs = [o1, o2]

    with pytest.raises(ValueError, match="No data"):
        ms.match(obs=obs, mod=mod)

    cc = ms.match(obs=obs, mod=mod, obs_no_overlap="ignore")
    assert "O1" in cc
    assert "O2" not in cc

    with pytest.warns(UserWarning, match="No data"):
        cc = ms.match(obs=obs, mod=mod, obs_no_overlap="warn")
    assert "O1" in cc
    assert "O2" not in cc


def test_score_changes_when_weights_override_defaults():
    time = pd.date_range("2000", periods=2)
    cc = ms.match(
        obs=[
            ms.PointObservation(
                pd.Series([2.0, 2.0], index=time),
                name="foo",
                weight=10.0,
            ),
            ms.PointObservation(pd.Series([1.0, 1.0], index=time), name="bar"),
        ],
        mod=ms.PointModelResult(pd.Series([0.0, 0.0], index=time), name="m"),
    )

    assert cc.score()["m"] == pytest.approx(1.90909)
    assert cc.score(weights={"bar": 2.0})["m"] == pytest.approx(1.8333333)
    assert cc.score(weights={"foo": 1.0, "bar": 2.0})["m"] == pytest.approx(1.333333)


def test_collection_has_copies_not_references_to_comparers():
    cmp1 = ms.from_matched(
        pd.DataFrame({"foo": [0, 0], "m1": [1, 1]}),
    )
    cmp2 = ms.from_matched(
        pd.DataFrame({"bar": [0, 0], "m1": [1, 1]}),
    )
    cc = ms.ComparerCollection([cmp1, cmp2])
    # modify the first comparer
    cc[0].data["m1"].attrs["random"] = "value"
    assert cc[0].data["m1"].attrs["random"] == "value"

    # cmp1 is unchanged
    assert "random" not in cmp1.data["m1"].attrs

    # the second comparer should not have this attribute
    assert "random" not in cc[1].data["m1"].attrs
import pytest
import numpy as np
import pandas as pd
import xarray as xr

import modelskill as ms
from modelskill.metrics import root_mean_squared_error, mean_absolute_error
from modelskill.comparison import Comparer


@pytest.fixture
def klagshamn():
    fn = "tests/testdata/smhi_2095_klagshamn.dfs0"
    return ms.PointObservation(fn, item=0, x=366844, y=6154291, name="Klagshamn")


@pytest.fixture
def drogden():
    fn = "tests/testdata/dmi_30357_Drogden_Fyr.dfs0"
    return ms.PointObservation(
        fn,
        item=0,
        x=355568.0,
        y=6156863.0,
        quantity=ms.Quantity("Water Level", unit="meter"),
    )


@pytest.fixture
def modelresult_oresund_WL():
    return ms.model_result("tests/testdata/Oresund2D_subset.dfsu", item=0)


@pytest.fixture
def cc(modelresult_oresund_WL, klagshamn, drogden):
    mr = modelresult_oresund_WL
    return ms.match([klagshamn, drogden], mr)


def test_get_comparer_by_name(cc):
    assert len(cc) == 2
    assert "Klagshamn" in cc.keys()
    assert "dmi_30357_Drogden_Fyr" in cc.keys()
    assert "Atlantis" not in cc.keys()


def test_get_comparer_by_position(cc):
    cc0 = cc[0]
    assert isinstance(cc0, Comparer)
    assert cc0.name == "Klagshamn"

    cc1 = cc[-1]
    assert isinstance(cc1, Comparer)
    assert cc1.name == "dmi_30357_Drogden_Fyr"

    ccs = cc[0:2]
    assert len(ccs) == 2
    assert "Klagshamn" in ccs


def test_subset_cc_for_named_comparers(cc):
    cmp = cc["Klagshamn"]
    assert cmp.name == "Klagshamn"

    cmp2 = cc[0]
    assert cmp2.name == "Klagshamn"

    ccs = cc[("Klagshamn", "dmi_30357_Drogden_Fyr")]
    assert len(ccs) == 2
    repr_text = repr(ccs)
    assert "<ComparerCollection>" in repr_text
    assert "Klagshamn" in repr_text
    assert "dmi_30357_Drogden_Fyr" in repr_text

    ccs2 = cc[["dmi_30357_Drogden_Fyr", "Klagshamn"]]
    repr_text = repr(ccs2)
    assert len(ccs2)
    assert "<ComparerCollection>" in repr_text
    assert "Klagshamn" in repr_text
    assert "dmi_30357_Drogden_Fyr" in repr_text


def test_iterate_over_comparers(cc):
    assert len(cc) == 2
    for c in cc:
        assert isinstance(c, Comparer)


def test_skill_from_observation_with_missing_values(modelresult_oresund_WL):
    o1 = ms.PointObservation(
        "tests/testdata/eq_ts_with_gaps.dfs0",
        item=0,
        x=366844,
        y=6154291,
        name="Klagshamn",
    )
    mr = modelresult_oresund_WL
    cmp = ms.match(o1, mr)
    df = cmp.skill().to_dataframe()
    assert not np.any(np.isnan(df))


def test_score_two_elements():
    mr = ms.model_result("tests/testdata/two_elements.dfsu", item=0)

    obs_df = pd.DataFrame(
        [2.0, 2.0], index=pd.date_range("2020-01-01", periods=2, freq="D")
    )

    # observation is in the center of the second element
    obs = ms.PointObservation(obs_df, item=0, x=2.0, y=2.0, name="obs")

    cc = ms.match(obs, mr, spatial_method="contained")

    assert cc.score(metric=root_mean_squared_error)["two_elements"] == pytest.approx(
        0.0
    )

    cc_default = ms.match(obs, mr)

    assert cc_default.score(metric=root_mean_squared_error)[
        "two_elements"
    ] == pytest.approx(0.0)


def test_score(modelresult_oresund_WL, klagshamn, drogden):
    mr = modelresult_oresund_WL

    cc = ms.match([klagshamn, drogden], mr)

    assert cc.score(metric=root_mean_squared_error)[
        "Oresund2D_subset"
    ] == pytest.approx(0.198637164895926)
    sk = cc.skill(metrics=[root_mean_squared_error, mean_absolute_error])
    sk.root_mean_squared_error.data.mean() == pytest.approx(0.198637164895926)


def test_weighted_score(modelresult_oresund_WL):
    o1 = ms.PointObservation(
        "tests/testdata/smhi_2095_klagshamn.dfs0",
        item=0,
        x=366844,
        y=6154291,
        name="Klagshamn",
    )
    o2 = ms.PointObservation(
        "tests/testdata/dmi_30357_Drogden_Fyr.dfs0",
        item=0,
        x=355568.0,
        y=6156863.0,
        quantity=ms.Quantity(
            "Water Level", unit="meter"
        ),  # not sure if this is relevant in this test
    )

    mr = ms.model_result("tests/testdata/Oresund2D_subset.dfsu", item=0, name="Oresund")

    cc = ms.match(obs=[o1, o2], mod=mr, spatial_method="contained")
    unweighted = cc.score()
    assert unweighted["Oresund"] == pytest.approx(0.1986296)

    # Weighted

    o1_w = ms.PointObservation(
        "tests/testdata/smhi_2095_klagshamn.dfs0",
        item=0,
        x=366844,
        y=6154291,
        name="Klagshamn",
        weight=10.0,
    )

    o2_w = ms.PointObservation(
        "tests/testdata/dmi_30357_Drogden_Fyr.dfs0",
        item=0,
        x=355568.0,
        y=6156863.0,
        quantity=ms.Quantity(
            "Water Level", unit="meter"
        ),  # not sure if this is relevant in this test
        weight=0.1,
    )

    cc_w = ms.match(obs=[o1_w, o2_w], mod=mr, spatial_method="contained")
    weighted = cc_w.score()

    assert weighted["Oresund"] == pytest.approx(0.1666888)


def test_weighted_score_from_prematched():
    df = pd.DataFrame(
        {"Oresund": [0.0, 1.0], "klagshamn": [0.0, 1.0], "drogden": [-1.0, 2.0]}
    )

    cmp1 = ms.from_matched(
        df[["Oresund", "klagshamn"]],
        mod_items=["Oresund"],
        obs_item="klagshamn",
        weight=100.0,
    )
    cmp2 = ms.from_matched(
        df[["Oresund", "drogden"]],
        mod_items=["Oresund"],
        obs_item="drogden",
        weight=0.0,
    )
    assert cmp1.weight == 100.0
    assert cmp2.weight == 0.0
    assert cmp1.score()["Oresund"] == pytest.approx(0.0)
    assert cmp2.score()["Oresund"] == pytest.approx(1.0)

    cc = ms.ComparerCollection([cmp1, cmp2])
    assert cc["klagshamn"].weight == 100.0
    assert cc["drogden"].weight == 0.0

    assert cc.score()["Oresund"] == pytest.approx(0.0)  # 100 * 0 + 0 * 1


def test_misc_properties(klagshamn, drogden):
    mr = ms.model_result("tests/testdata/Oresund2D_subset.dfsu", item=0)

    cc = ms.match([klagshamn, drogden], mr)

    assert len(cc) == 2

    assert cc.n_models == 1
    assert cc.mod_names == ["Oresund2D_subset"]

    ck = cc["Klagshamn"]
    assert ck.name == "Klagshamn"

    assert ck.n_points > 0

    assert ck.time[0].year == 2018  # intersection of observation and model times
    assert ck.time[-1].year == 2018

    assert ck.x == 366844


# def test_sel_time(cc):
#     c1 = cc["Klagshamn"]
#     c2 = c1.sel(time=slice("2018-01-01", "2018-01-02"))
#     assert c2.start == datetime(2018, 1, 1)


def test_skill(klagshamn, drogden):
    mr = ms.model_result("tests/testdata/Oresund2D_subset.dfsu", item=0)

    cc = ms.match([klagshamn, drogden], mr)

    df = cc.skill().to_dataframe()
    assert df.loc["Klagshamn"].n == 71

    # Filtered skill
    df = cc.sel(observation="Klagshamn").skill().to_dataframe()
    assert df.loc["Klagshamn"].n == 71


def test_skill_choose_metrics(klagshamn, drogden):
    mr = ms.model_result("tests/testdata/Oresund2D_subset.dfsu", item=0)

    cc = ms.match([klagshamn, drogden], mr)

    df = cc.skill(metrics=["mae", "si"]).to_dataframe()

    assert "mae" in df.columns
    assert "rmse" not in df.columns

    # Override defaults
    df = cc.skill(metrics=["urmse", "r2"]).to_dataframe()

    assert "r2" in df.columns
    assert "rmse" not in df.columns


def test_skill_choose_metrics_back_defaults(cc):
    df = cc.skill(metrics=["kge", "nse", "max_error"]).to_dataframe()
    assert "kge" in df.columns
    assert "rmse" not in df.columns

    df = cc.mean_skill(metrics=["kge", "nse", "max_error"]).to_dataframe()
    assert "kge" in df.columns
    assert "rmse" not in df.columns

    df = cc.mean_skill().to_dataframe()
    assert "kge" not in df.columns
    assert "rmse" in df.columns


def test_obs_attrs_carried_over(klagshamn, modelresult_oresund_WL):
    klagshamn.data.attrs["A"] = "B"  # could also have been added in constructor
    cmp = ms.match(klagshamn, modelresult_oresund_WL)
    assert cmp.data.attrs["A"] == "B"


def test_obs_aux_carried_over(klagshamn, modelresult_oresund_WL):
    klagshamn.data["aux"] = xr.ones_like(klagshamn.data["Klagshamn"])
    klagshamn.data["aux"].attrs["kind"] = "aux"
    cmp = ms.match(klagshamn, modelresult_oresund_WL)
    assert "aux" in cmp.data
    assert cmp.data["aux"].values[0] == 1.0
    assert cmp.data["aux"].attrs["kind"] == "aux"
    assert cmp.mod_names == ["Oresund2D_subset"]


def test_obs_aux_carried_over_nan(klagshamn, modelresult_oresund_WL):
    cmp1 = ms.match(klagshamn, modelresult_oresund_WL)
    assert cmp1.n_points == 71
    assert cmp1.time[0] == pd.Timestamp("2018-03-04 00:00:00")
    assert cmp1.data["Observation"].values[0] == pytest.approx(-0.11)

    # NaN values in aux should not influence the comparison
    klagshamn.data["aux"] = xr.ones_like(klagshamn.data["Klagshamn"])
    klagshamn.data["aux"].attrs["kind"] = "aux"
    klagshamn.data["aux"].loc["2018-03-04 00:00:00"] = np.nan
    cmp2 = ms.match(klagshamn, modelresult_oresund_WL)
    assert cmp2.n_points == 71
    assert cmp2.time[0] == pd.Timestamp("2018-03-04 00:00:00")
    assert cmp2.data["Observation"].values[0] == pytest.approx(-0.11)


def test_mod_aux_carried_over(klagshamn):
    mr = ms.model_result(
        "tests/testdata/Oresund2D_subset.dfsu", item=0, aux_items="U velocity"
    )
    cmp = ms.match(klagshamn, mr, spatial_method="contained")
    assert "U velocity" in cmp.data.data_vars
    assert cmp.data["U velocity"].values[0] == pytest.approx(-0.0360998)
    assert cmp.data["U velocity"].attrs["kind"] == "aux"
    assert cmp.mod_names == ["Oresund2D_subset"]
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.table import Table

import modelskill as ms
import modelskill.metrics as mtr

plt.rcParams.update({"figure.max_open_warning": 0})


@pytest.fixture
def mr1():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    return ms.model_result(fn, item=0, name="SW_1")


@pytest.fixture
def mr2():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    return ms.model_result(fn, item=0, name="SW_2")


@pytest.fixture
def o1():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return ms.PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def o2():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return ms.PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def o3():
    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    return ms.TrackObservation(fn, item=3, name="c2")


@pytest.fixture
def cc(mr1, mr2, o1, o2, o3):
    return ms.match([o1, o2, o3], [mr1, mr2], spatial_method="nearest")


def test_compare(mr1, mr2, o1, o2, o3):
    cc = ms.match([o1, o2, o3], [mr1, mr2])

    assert cc.n_points > 0
    assert "ComparerCollection" in repr(cc)
    assert "Comparer" in repr(cc["EPL"])
    assert "Comparer" in repr(cc[2])


def test_merge_comparer(mr1, mr2, o1, o2):
    cc1 = ms.match(o1, mr1)
    cc2 = ms.match(o2, mr2)
    cc = cc1.merge(cc2)
    assert cc.n_points > 0
    assert "ComparerCollection" in repr(cc)
    assert "Comparer" in repr(cc["EPL"])
    assert "Comparer" in repr(cc["HKNA"])


def test_add_same_comparer_twice(mr1, mr2, o1, o2):
    cc1 = ms.match(o1, mr1)
    cc2 = ms.match(o2, mr2)
    cc = cc1.merge(cc2)
    assert len(cc) == 2
    cc = cc.merge(cc2)
    assert len(cc) == 2  # adding the same comparer again doesn't have any effect
    assert cc.n_points > 0
    assert "ComparerCollection" in repr(cc)
    assert "Comparer" in repr(cc["EPL"])
    assert "Comparer" in repr(cc["HKNA"])


def test_mm_skill(cc):
    df = cc.sel(start="2017-10-27 00:01").skill().to_dataframe()

    # mod: ['SW_1', 'SW_2'], obs: ['HKNA', 'EPL', 'c2']
    assert df.iloc[3].name[0] == "SW_2"
    assert df.iloc[3].name[1] == "HKNA"
    assert pytest.approx(df.iloc[3].mae, 1e-5) == 0.214476

    assert df.iloc[3].name[0] == "SW_2"
    assert df.iloc[3].name[1] == "HKNA"
    assert pytest.approx(df.iloc[3].mae, 1e-5) == 0.214476


def test_mm_skill_model(cc):
    df = cc.sel(model="SW_1").skill().to_dataframe()
    assert df.loc["EPL"].n == 67
    assert df.loc["c2"].n == 113

    df2 = cc.sel(model=-2).skill().to_dataframe()
    assert df2.loc["c2"].rmse == df.loc["c2"].rmse


def test_mm_sel_missing_model(cc):
    with pytest.raises(KeyError):
        cc.sel(model="SW_3")
    with pytest.raises(IndexError):
        cc.sel(model=999)
    with pytest.raises((KeyError, IndexError)):
        cc.sel(model=[999, "SW_2"])
    with pytest.raises(TypeError):
        cc.sel(model=[0.1])


def test_mm_skill_obs(cc):
    sk = cc.sel(observation="c2").skill()
    assert len(sk) == 2
    assert pytest.approx(sk.loc["SW_2"].bias) == 0.081431053

    sk2 = cc.sel(observation=-1).skill()
    assert pytest.approx(sk2.loc["SW_2"].bias) == 0.081431053


def test_mm_mean_skill_obs(cc):
    df = cc.sel(model=0, observation=[0, "c2"]).mean_skill().to_dataframe()
    assert pytest.approx(df.iloc[0].si) == 0.11113215


def test_mm_sel_missing_obs(cc, o1):
    with pytest.raises(KeyError):
        cc.sel(observation="imaginary_obs")
    with pytest.raises(IndexError):
        cc.sel(observation=999)
    with pytest.raises((KeyError, IndexError)):
        cc.sel(observation=["c2", 999])
    with pytest.raises(TypeError):
        cc.sel(observation=[o1])


def test_mm_skill_start_end(cc):
    # TODO should we keep these tests?
    sk = cc.sel(model="SW_1", start="2017").skill()
    assert sk.loc["EPL"].n == 67
    sk = cc.sel(model="SW_1", end="2017-10-28 00:00:00").skill()
    assert sk.loc["EPL"].n == 25
    sk = cc.sel(model="SW_1", start="2017-10-28 00:00:01").skill()
    assert sk.loc["EPL"].n == 42


def test_mm_skill_area_bbox(cc):
    bbox = [0.5, 52.5, 5, 54]
    sk = cc.sel(model="SW_1", area=bbox).skill()
    assert pytest.approx(sk.loc["HKNA"].urmse) == 0.293498777
    bbox = np.array([0.5, 52.5, 5, 54])
    sk = cc.sel(model="SW_1", area=bbox).skill()
    assert pytest.approx(sk.loc["HKNA"].urmse) == 0.293498777


def test_mm_skill_area_polygon(cc):
    polygon = np.array([[6, 51], [0, 55], [0, 51], [6, 51]])
    sk = cc.sel(model="SW_2", area=polygon).skill()
    assert "HKNA" not in sk.obs_names
    assert sk.to_dataframe().iloc[1].n == 66

    # "this is not the indexing you want..."
    # assert pytest.approx(s.iloc[0].r2) == 0.9271339372

    # same as above but not closed
    polygon = np.array([[6, 51], [0, 55], [0, 51]])
    sk = cc.sel(model="SW_2", area=polygon).skill()

    # assert pytest.approx(s.iloc[0].r2) == 0.9271339372

    polygon = [6, 51, 0, 55, 0, 51, 6, 51]
    sk = cc.sel(model="SW_2", area=polygon).skill()
    # assert pytest.approx(s.iloc[0].r2) == 0.9271339372

    # same as above but not closed
    polygon = [6, 51, 0, 55, 0, 51]
    sk = cc.sel(model="SW_2", area=polygon).skill()
    # assert pytest.approx(s.iloc[0].r2) == 0.9271339372


def test_mm_mean_skill_area_polygon(cc):
    # The OGC standard definition requires a polygon to be topologically closed.
    # It also states that if the exterior linear ring of a polygon is defined in a counterclockwise direction, then it will be seen from the "top".
    # Any interior linear rings should be defined in opposite fashion compared to the exterior ring, in this case, clockwise
    polygon = np.array([[6, 51], [0, 55], [0, 51], [6, 51]])
    sk = cc.sel(area=polygon).mean_skill()
    assert pytest.approx(sk.loc["SW_2"].rmse) == 0.3349027897

    closed_polygon = ((6, 51), (0, 55), (0, 51), (6, 51))
    sk2 = cc.sel(area=closed_polygon).mean_skill()
    assert pytest.approx(sk2.loc["SW_2"].rmse) == 0.3349027897

    # TODO support for polygons with holes


def test_mm_sel_area_error(cc):
    with pytest.raises(ValueError):
        cc.sel(area=[0.1, 0.2])
    with pytest.raises(ValueError):
        cc.sel(area="polygon")
    with pytest.raises(ValueError):
        cc.sel(area=[0.1, 0.2, 0.3, 0.6, "string"])
    with pytest.raises(ValueError):
        # uneven number of elements
        cc.sel(area=[0.1, 0.2, 0.3, 0.6, 5.6, 5.9, 5.0])
    with pytest.raises(ValueError):
        polygon = np.array([[6, 51, 4], [0, 55, 4], [0, 51, 4], [6, 51, 4]])
        cc.sel(area=polygon)


def test_mm_skill_metrics(cc):
    df = cc.sel(model="SW_1").skill(metrics=[mtr.mean_absolute_error]).to_dataframe()
    assert df.mean_absolute_error.values.sum() > 0.0

    sk = cc.sel(model="SW_1").skill(metrics=[mtr.bias, "rmse"])
    assert pytest.approx(sk.loc["EPL"].bias) == -0.06659714
    assert pytest.approx(sk.loc["EPL"].rmse) == 0.22359664

    with pytest.raises(ValueError):
        cc.sel(model="SW_1").skill(metrics=["mean_se"])
    with pytest.raises(AttributeError):
        cc.sel(model="SW_1").skill(metrics=[mtr.fake])
    with pytest.raises(TypeError):
        cc.sel(model="SW_1").skill(metrics=[47])


def test_mm_mean_skill(cc):
    sk = cc.mean_skill()
    assert len(sk) == 2
    assert sk.loc["SW_1"].rmse == pytest.approx(0.309118939)


def test_mm_mean_skill_weights_list(cc):
    sk = cc.mean_skill(weights=[0.2, 0.3, 1.0])
    assert len(sk) == 2
    assert sk.loc["SW_1"].rmse == pytest.approx(0.3261788143)

    sk = cc.mean_skill(weights=[100000000000.0, 1.0, 1.0])
    assert sk.loc["SW_1"].rmse < 1.0

    # sk = cc.mean_skill(weights=1)
    # assert len(sk) == 2
    # assert sk.loc["SW_1"].rmse == pytest.approx(0.309118939)

    with pytest.raises(ValueError):
        # too many weights
        cc.mean_skill(weights=[0.2, 0.3, 0.4, 0.5])


def test_mm_mean_skill_weights_str(cc):
    sk = cc.mean_skill(weights="points")
    assert len(sk) == 2
    assert sk.loc["SW_1"].rmse == pytest.approx(0.3367349)

    sk = cc.mean_skill(weights="equal")
    assert len(sk) == 2
    assert sk.loc["SW_1"].rmse == pytest.approx(0.309118939)


def test_mm_mean_skill_weights_dict(cc):
    sk = cc.mean_skill(weights={"EPL": 0.3, "c2": 1.0, "HKNA": 0.2})
    df = sk.to_dataframe()
    assert len(sk) == 2
    assert df.loc["SW_1"].rmse == pytest.approx(0.3261788143)

    # s2 = cc.mean_skill(weights=[0.3, 0.2, 1.0])

    # TODO this is not a good way to test
    # assert s.loc["SW_1"].rmse == s2.loc["SW_1"].rmse
    # assert s.loc["SW_2"].rmse == s2.loc["SW_2"].rmse

    df = cc.mean_skill(weights={"HKNA": 2.0}).to_dataframe()
    assert len(sk) == 2
    assert df.loc["SW_1"].rmse == pytest.approx(0.319830126)

    # df2 = cc.mean_skill(weights={"EPL": 2.0, "c2": 1.0, "HKNA": 1.0}).to_dataframe()

    # TODO asserts with hard-coded expected values
    # assert s.loc["SW_1"].rmse == s2.loc["SW_1"].rmse
    # assert s.loc["SW_2"].rmse == s2.loc["SW_2"].rmse


# TODO: mean_skill_points needs fixing before this test can be enabled
# def test_mean_skill_points(cc):
#     sk = cc.mean_skill_points()
#     assert len(sk) == 2
#     assert sk.loc["SW_1"].rmse == pytest.approx(0.33927729)


def test_mm_scatter(cc):
    # scatter is the default plot
    ax = cc.sel(model="SW_2").plot()
    assert "SW_2" in ax.get_title()

    cc.sel(model="SW_1", observation=[0, 1]).plot.scatter()
    cc.sel(model="SW_2").plot.scatter(show_points=False)
    cc.sel(model="SW_2").plot.scatter(show_hist=False)
    cc.sel(model="SW_2").plot.scatter(bins=0.5)
    cc.sel(model="SW_2").plot.scatter(title="t", xlabel="x", ylabel="y")
    cc.sel(model="SW_2").plot.scatter(show_points=True)
    cc.sel(model="SW_2").plot.scatter(show_points=100)
    cc.sel(model="SW_2").plot.scatter(show_points=0.75)
    cc.sel(model="SW_2").plot.scatter(show_density=True)
    cc.sel(model="SW_2").plot.scatter(show_points=0.75, show_density=True)
    cc.sel(model="SW_2", observation="HKNA").plot.scatter(skill_table=True)
    cc.sel(model="SW_2").plot.scatter(fit_to_quantiles=True)
    # cc.sel(model="SW_2").plot.scatter(binsize=0.5, backend="plotly")
    assert True
    plt.close("all")


def cm_1(obs, model):
    """Custom metric #1"""
    return np.mean(obs / model)


def cm_2(obs, model):
    """Custom metric #2"""
    return np.mean(obs * 1.5 / model)


def cm_3(obs, model):
    """Custom metric #3"""
    return 42


def test_custom_metric_skilltable_mm_scatter(cc):
    mtr.add_metric(cm_1)
    mtr.add_metric(cm_2, has_units=True)
    ccs = cc.sel(model="SW_2", observation="HKNA")
    ccs.plot.scatter(skill_table=["bias", cm_1, "si", cm_2])
    assert True
    plt.close("all")

    mtr.add_metric(cm_1)

    assert mtr.is_valid_metric("cm_1")

    # use custom metric as function
    sk = cc.skill(metrics=[cm_1])
    assert sk["cm_1"] is not None

    # use custom metric as string
    cc.skill(metrics=["cm_1"])
    assert sk["cm_1"] is not None

    # using a non-registred metric raises an error, since it cannot be found in the registry
    with pytest.raises(ValueError) as e_info:
        cc.skill(metrics=["cm_3"])
    assert "add_metric" in str(e_info.value)

    # using it as a function directly is ok
    cc.skill(metrics=[cm_3])


def test_mm_kde(cc):
    ax = cc.sel(model="SW_2").plot.kde()
    assert ax is not None
    # TODO more informative test


def test_mm_hist(cc):
    ax = cc.sel(model="SW_2").plot.hist()
    assert ax is not None


def test_mm_taylor(cc):
    cc.sel(model="SW_1", observation=[0, 1]).plot.taylor()
    cc.sel(model="SW_2").plot.taylor(normalize_std=True)
    cc.sel(model="SW_2").plot.taylor(figsize=(4, 4))
    cc.sel(model="SW_2", start="2017-10-28").plot.taylor()
    cc[0].sel(model=0, end="2017-10-29").plot.taylor()
    assert True
    plt.close("all")


def test_mm_plot_timeseries(cc):
    cc["EPL"].plot.timeseries()
    cc["EPL"].plot.timeseries(title="t", figsize=(3, 3))

    # cc["EPL"].plot_timeseries(backend="plotly")
    with pytest.raises(ValueError):
        cc["EPL"].plot.timeseries(backend="mpl")

    ax = cc["EPL"].plot.timeseries()
    assert "EPL" in ax.get_title()

    plt.close("all")


def test_match_including_dummy(mr1, mr2, o1, o2, o3):
    mr3 = ms.DummyModelResult(strategy="constant", data=0.0)
    cc = ms.match([o1, o2, o3], [mr3, mr1, mr2])
    assert "dummy" in cc.mod_names
