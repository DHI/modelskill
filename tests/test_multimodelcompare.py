import matplotlib.pyplot as plt
import numpy as np
import pytest

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
    return ms.match([o1, o2, o3], [mr1, mr2])


def test_compare(mr1, mr2, o1, o2, o3):
    cc = ms.match([o1, o2, o3], [mr1, mr2])

    assert cc.n_points > 0
    assert "ComparerCollection" in repr(cc)
    assert "Comparer" in repr(cc["EPL"])
    assert "Comparer" in repr(cc[2])


def test_add_comparer(mr1, mr2, o1, o2):
    cc1 = ms.match(o1, mr1)
    cc2 = ms.match(o2, mr2)
    cc = cc1 + cc2
    assert cc.n_points > 0
    assert "ComparerCollection" in repr(cc)
    assert "Comparer" in repr(cc["EPL"])
    assert "Comparer" in repr(cc["HKNA"])


def test_add_same_comparer_twice(mr1, mr2, o1, o2):
    cc1 = ms.match(o1, mr1)
    cc2 = ms.match(o2, mr2)
    cc = cc1 + cc2
    assert len(cc) == 2
    cc = cc + cc2
    assert len(cc) == 2  # adding the same comparer again doesn't have any effect
    assert cc.n_points > 0
    assert "ComparerCollection" in repr(cc)
    assert "Comparer" in repr(cc["EPL"])
    assert "Comparer" in repr(cc["HKNA"])


def test_mm_skill(cc):
    df = cc.sel(start="2017-10-27 00:01").skill().to_dataframe()

    assert df.iloc[4].name[0] == "SW_2"
    assert df.iloc[4].name[1] == "HKNA"
    assert pytest.approx(df.iloc[4].mae, 1e-5) == 0.214476

    # TODO remove in v1.1
    with pytest.warns(FutureWarning):
        df = cc.skill(start="2017-10-27 00:01").to_dataframe()

    assert df.iloc[4].name[0] == "SW_2"
    assert df.iloc[4].name[1] == "HKNA"
    assert pytest.approx(df.iloc[4].mae, 1e-5) == 0.214476


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
    with pytest.warns(FutureWarning):
        s = cc.skill(observation="c2")
        assert len(s) == 2
        assert pytest.approx(s.loc["SW_2"].bias) == 0.081431053

    s = cc.sel(observation="c2").skill()
    assert len(s) == 2
    assert pytest.approx(s.loc["SW_2"].bias) == 0.081431053

    s2 = cc.sel(observation=-1).skill()
    assert s.loc["SW_2"].bias == s2.loc["SW_2"].bias


def test_mm_isel(cc):
    s2 = cc.isel(observation=-1).skill()
    assert s2.loc["SW_2"].bias == pytest.approx(0.081431053)


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
    s = cc.sel(model="SW_1", start="2017").skill()
    assert s.loc["EPL"].n == 67
    s = cc.sel(model="SW_1", end="2017-10-28 00:00:00").skill()
    assert s.loc["EPL"].n == 25
    s = cc.sel(model="SW_1", start="2017-10-28 00:00:01").skill()
    assert s.loc["EPL"].n == 42


def test_mm_skill_area_bbox(cc):
    bbox = [0.5, 52.5, 5, 54]
    s = cc.sel(model="SW_1", area=bbox).skill()
    assert pytest.approx(s.loc["HKNA"].urmse) == 0.293498777
    bbox = np.array([0.5, 52.5, 5, 54])
    s = cc.sel(model="SW_1", area=bbox).skill()
    assert pytest.approx(s.loc["HKNA"].urmse) == 0.293498777


def test_mm_skill_area_polygon(cc):
    polygon = np.array([[6, 51], [0, 55], [0, 51], [6, 51]])
    s = cc.sel(model="SW_2", area=polygon).skill()
    assert "HKNA" not in s.obs_names
    assert s.to_dataframe().iloc[1].n == 66

    # "this is not the indexing you want..."
    # assert pytest.approx(s.iloc[0].r2) == 0.9271339372

    # same as above but not closed
    polygon = np.array([[6, 51], [0, 55], [0, 51]])
    s = cc.sel(model="SW_2", area=polygon).skill()

    # assert pytest.approx(s.iloc[0].r2) == 0.9271339372

    polygon = [6, 51, 0, 55, 0, 51, 6, 51]
    s = cc.sel(model="SW_2", area=polygon).skill()
    # assert pytest.approx(s.iloc[0].r2) == 0.9271339372

    # same as above but not closed
    polygon = [6, 51, 0, 55, 0, 51]
    s = cc.sel(model="SW_2", area=polygon).skill()
    # assert pytest.approx(s.iloc[0].r2) == 0.9271339372


def test_mm_mean_skill_area_polygon(cc):
    # The OGC standard definition requires a polygon to be topologically closed.
    # It also states that if the exterior linear ring of a polygon is defined in a counterclockwise direction, then it will be seen from the "top".
    # Any interior linear rings should be defined in opposite fashion compared to the exterior ring, in this case, clockwise
    polygon = np.array([[6, 51], [0, 55], [0, 51], [6, 51]])
    s = cc.sel(area=polygon).mean_skill()
    assert pytest.approx(s.loc["SW_2"].rmse) == 0.3349027897

    closed_polygon = ((6, 51), (0, 55), (0, 51), (6, 51))
    s2 = cc.sel(area=closed_polygon).mean_skill()
    assert pytest.approx(s2.loc["SW_2"].rmse) == 0.3349027897

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

    s = cc.sel(model="SW_1").skill(metrics=[mtr.bias, "rmse"])
    assert pytest.approx(s.loc["EPL"].bias) == -0.06659714
    assert pytest.approx(s.loc["EPL"].rmse) == 0.22359664

    with pytest.raises(ValueError):
        cc.sel(model="SW_1").skill(metrics=["mean_se"])
    with pytest.raises(AttributeError):
        cc.sel(model="SW_1").skill(metrics=[mtr.fake])
    with pytest.raises(TypeError):
        cc.sel(model="SW_1").skill(metrics=[47])


def test_mm_mean_skill(cc):
    s = cc.mean_skill()
    assert len(s) == 2
    assert s.loc["SW_1"].rmse == pytest.approx(0.309118939)


def test_mm_mean_skill_weights_list(cc):
    s = cc.mean_skill(weights=[0.3, 0.2, 1.0])
    assert len(s) == 2
    assert s.loc["SW_1"].rmse == pytest.approx(0.3261788143)

    s = cc.mean_skill(weights=[100000000000.0, 1.0, 1.0])
    assert s.loc["SW_1"].rmse < 1.0

    s = cc.mean_skill(weights=1)
    assert len(s) == 2
    assert s.loc["SW_1"].rmse == pytest.approx(0.309118939)

    with pytest.raises(ValueError):
        # too many weights
        cc.mean_skill(weights=[0.2, 0.3, 0.4, 0.5])


def test_mm_mean_skill_weights_str(cc):
    s = cc.mean_skill(weights="points")
    assert len(s) == 2
    assert s.loc["SW_1"].rmse == pytest.approx(0.3367349)

    s = cc.mean_skill(weights="equal")
    assert len(s) == 2
    assert s.loc["SW_1"].rmse == pytest.approx(0.309118939)


def test_mm_mean_skill_weights_dict(cc):
    s = cc.mean_skill(weights={"EPL": 0.2, "c2": 1.0, "HKNA": 0.3})
    df = s.to_dataframe()
    assert len(s) == 2
    assert df.loc["SW_1"].rmse == pytest.approx(0.3261788143)

    # s2 = cc.mean_skill(weights=[0.3, 0.2, 1.0])

    # TODO this is not a good way to test
    # assert s.loc["SW_1"].rmse == s2.loc["SW_1"].rmse
    # assert s.loc["SW_2"].rmse == s2.loc["SW_2"].rmse

    df = cc.mean_skill(weights={"EPL": 2.0}).to_dataframe()
    assert len(s) == 2
    assert df.loc["SW_1"].rmse == pytest.approx(0.319830126)

    # df2 = cc.mean_skill(weights={"EPL": 2.0, "c2": 1.0, "HKNA": 1.0}).to_dataframe()

    # TODO asserts with hard-coded expected values
    # assert s.loc["SW_1"].rmse == s2.loc["SW_1"].rmse
    # assert s.loc["SW_2"].rmse == s2.loc["SW_2"].rmse


# TODO: mean_skill_points needs fixing before this test can be enabled
# def test_mean_skill_points(cc):
#     s = cc.mean_skill_points()
#     assert len(s) == 2
#     assert s.loc["SW_1"].rmse == pytest.approx(0.33927729)


def test_mm_scatter(cc):
    with pytest.warns(FutureWarning):
        cc.sel(model="SW_2").scatter(start="2017-10-28")

    with pytest.warns(FutureWarning):
        cc.sel(model="SW_2")[0].scatter(start="2017-10-28")

    with pytest.warns(FutureWarning):
        cc.sel(model="SW_2").scatter()

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
    return np.mean(obs.ravel() / model.ravel())


def cm_2(obs, model):
    """Custom metric #2"""
    return np.mean(obs.ravel() * 1.5 / model.ravel())


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
    s = cc.skill(metrics=[cm_1])
    assert s["cm_1"] is not None

    # use custom metric as string
    cc.skill(metrics=["cm_1"])
    assert s["cm_1"] is not None

    # using a non-registred metric raises an error, even though it is a defined function, but not registered
    with pytest.raises(ValueError) as e_info:
        cc.skill(metrics=["cm_3"])

    assert "add_metric" in str(e_info.value)

    with pytest.raises(ValueError) as e_info:
        cc.skill(metrics=[cm_3])

    assert "add_metric" in str(e_info.value)


def test_mm_kde(cc):
    with pytest.warns(FutureWarning):
        cc.kde()

    ax = cc.sel(model="SW_2").plot.kde()
    assert ax is not None
    # TODO more informative test


def test_mm_hist(cc):
    ax = cc.sel(model="SW_2").plot.hist()
    assert ax is not None


def test_mm_taylor(cc):
    with pytest.warns(FutureWarning):
        cc.taylor()

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
