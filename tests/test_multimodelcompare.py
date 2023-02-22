import pytest
import numpy as np
import matplotlib.pyplot as plt

from fmskill import ModelResult
from fmskill import PointObservation, TrackObservation
from fmskill import Connector
import fmskill.metrics as mtr

plt.rcParams.update({"figure.max_open_warning": 0})


@pytest.fixture
def mr1():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    return ModelResult(fn, name="SW_1")


@pytest.fixture
def mr2():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    return ModelResult(fn, name="SW_2")


@pytest.fixture
def o1():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def o2():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def o3():
    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    return TrackObservation(fn, item=3, name="c2")


@pytest.fixture
def cc(mr1, mr2, o1, o2, o3):
    con = Connector([o1, o2, o3], [mr1[0], mr2[0]])
    return con.extract()


def test_connector(mr1, mr2, o1):
    con = Connector(o1, [mr1[0], mr2[0]])
    assert len(con.observations) == 1


def test_extract(mr1, mr2, o1, o2, o3):
    con = Connector([o1, o2, o3], [mr1[0], mr2[0]])
    cc = con.extract()

    assert cc.n_points > 0
    assert "ComparerCollection" in repr(cc)
    assert "PointComparer" in repr(cc["EPL"])
    assert "TrackComparer" in repr(cc[2])


def test_add_comparer(mr1, mr2, o1, o2):
    cc1 = Connector(o1, mr1[0]).extract()
    cc2 = Connector(o2, mr2[0]).extract()
    cc = cc1 + cc2
    assert cc.n_points > 0
    assert "ComparerCollection" in repr(cc)
    assert "PointComparer" in repr(cc["EPL"])
    assert "PointComparer" in repr(cc["HKNA"])


def test_add_same_comparer_twice(mr1, mr2, o1, o2):
    cc1 = Connector(o1, mr1[0]).extract()
    cc2 = Connector(o2, mr2[0]).extract()
    cc = cc1 + cc2
    assert len(cc) == 2
    cc = cc + cc2
    assert len(cc) == 2  # adding the same comparer again doesn't have any effect
    assert cc.n_points > 0
    assert "ComparerCollection" in repr(cc)
    assert "PointComparer" in repr(cc["EPL"])
    assert "PointComparer" in repr(cc["HKNA"])


def test_mm_skill(cc):
    df = cc.skill(start="2017-10-27 00:01").df
    assert df.iloc[4].name[0] == "SW_2"
    assert df.iloc[4].name[1] == "HKNA"
    assert pytest.approx(df.iloc[4].mae, 1e-5) == 0.214476


def test_mm_skill_model(cc):
    df = cc.skill(model="SW_1").df
    assert df.loc["EPL"].n == 67
    assert df.loc["c2"].n == 113

    df2 = cc.skill(model=-2).df
    assert df2.loc["c2"].rmse == df.loc["c2"].rmse


def test_mm_skill_missing_model(cc):
    with pytest.raises(KeyError):
        cc.skill(model="SW_3")
    with pytest.raises(IndexError):
        cc.skill(model=999)
    with pytest.raises((KeyError, IndexError)):
        cc.skill(model=[999, "SW_2"])
    with pytest.raises(TypeError):
        cc.skill(model=[0.1])


def test_mm_skill_obs(cc):
    s = cc.skill(observation="c2")
    assert len(s) == 2
    assert pytest.approx(s.loc["SW_2"].bias) == 0.081431053

    s2 = cc.skill(observation=-1)
    assert s.loc["SW_2"].bias == s2.loc["SW_2"].bias

    df = cc.mean_skill(model=0, observation=[0, "c2"]).df
    assert pytest.approx(df.si[0]) == 0.11113215


def test_mm_skill_missing_obs(cc, o1):
    with pytest.raises(KeyError):
        cc.skill(observation="imaginary_obs")
    with pytest.raises(IndexError):
        cc.skill(observation=999)
    with pytest.raises((KeyError, IndexError)):
        cc.skill(observation=["c2", 999])
    with pytest.raises(TypeError):
        cc.skill(observation=[o1])


def test_mm_skill_start_end(cc):
    s = cc.skill(model="SW_1", start="2017")
    assert s.loc["EPL"].n == 67
    s = cc.skill(model="SW_1", end="2017-10-28 00:00:00")
    assert s.loc["EPL"].n == 25
    s = cc.skill(model="SW_1", start="2017-10-28 00:00:01")
    assert s.loc["EPL"].n == 42


def test_mm_skill_area_bbox(cc):
    bbox = [0.5, 52.5, 5, 54]
    s = cc.skill(model="SW_1", area=bbox)
    assert pytest.approx(s.loc["HKNA"].urmse) == 0.293498777
    bbox = np.array([0.5, 52.5, 5, 54])
    s = cc.skill(model="SW_1", area=bbox)
    assert pytest.approx(s.loc["HKNA"].urmse) == 0.293498777


def test_mm_skill_area_polygon(cc):
    polygon = np.array([[6, 51], [0, 55], [0, 51], [6, 51]])
    s = cc.skill(model="SW_2", area=polygon)
    assert "HKNA" not in s.index
    assert s.df.n[1] == 66
    assert pytest.approx(s.iloc[0].r2) == 0.9271339372

    # same as above but not closed
    polygon = np.array([[6, 51], [0, 55], [0, 51]])
    s = cc.skill(model="SW_2", area=polygon)
    assert pytest.approx(s.iloc[0].r2) == 0.9271339372

    polygon = [6, 51, 0, 55, 0, 51, 6, 51]
    s = cc.skill(model="SW_2", area=polygon)
    assert pytest.approx(s.iloc[0].r2) == 0.9271339372

    # same as above but not closed
    polygon = [6, 51, 0, 55, 0, 51]
    s = cc.skill(model="SW_2", area=polygon)
    assert pytest.approx(s.iloc[0].r2) == 0.9271339372

    s = cc.mean_skill(area=polygon)
    assert pytest.approx(s.loc["SW_2"].rmse) == 0.3349027897


def test_mm_skill_area_error(cc):
    with pytest.raises(ValueError):
        cc.skill(area=[0.1, 0.2])
    with pytest.raises(ValueError):
        cc.skill(area="polygon")
    with pytest.raises(ValueError):
        cc.skill(area=[0.1, 0.2, 0.3, 0.6, "string"])
    with pytest.raises(ValueError):
        # uneven number of elements
        cc.skill(area=[0.1, 0.2, 0.3, 0.6, 5.6, 5.9, 5.0])
    with pytest.raises(ValueError):
        polygon = np.array([[6, 51, 4], [0, 55, 4], [0, 51, 4], [6, 51, 4]])
        cc.skill(area=polygon)


def test_mm_skill_metrics(cc):
    df = cc.skill(model="SW_1", metrics=[mtr.mean_absolute_error]).df
    assert df.mean_absolute_error.values.sum() > 0.0

    s = cc.skill(model="SW_1", metrics=[mtr.bias, "rmse"])
    assert pytest.approx(s.loc["EPL"].bias) == -0.06659714
    assert pytest.approx(s.loc["EPL"].rmse) == 0.22359664

    with pytest.raises(ValueError):
        cc.skill(model="SW_1", metrics=["mean_se"])
    with pytest.raises(AttributeError):
        cc.skill(model="SW_1", metrics=[mtr.fake])
    with pytest.raises(TypeError):
        cc.skill(model="SW_1", metrics=[47])


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
    assert len(s) == 2
    assert s.loc["SW_1"].rmse == pytest.approx(0.3261788143)

    s2 = cc.mean_skill(weights=[0.3, 0.2, 1.0])
    assert s.loc["SW_1"].rmse == s2.loc["SW_1"].rmse
    assert s.loc["SW_2"].rmse == s2.loc["SW_2"].rmse

    s = cc.mean_skill(weights={"EPL": 2.0})
    assert len(s) == 2
    assert s.loc["SW_1"].rmse == pytest.approx(0.319830126)

    s2 = cc.mean_skill(weights={"EPL": 2.0, "c2": 1.0, "HKNA": 1.0})
    assert s.loc["SW_1"].rmse == s2.loc["SW_1"].rmse
    assert s.loc["SW_2"].rmse == s2.loc["SW_2"].rmse


def test_mean_skill_points(cc):
    s = cc.mean_skill_points()
    assert len(s) == 2
    assert s.loc["SW_1"].rmse == pytest.approx(0.33927729)


def test_mm_scatter(cc):
    cc.scatter(model="SW_1", observation=[0, 1])
    cc.scatter(model="SW_2", show_points=False)
    cc.scatter(model="SW_2", show_hist=False)
    cc.scatter(model="SW_2", bins=0.5)
    with pytest.warns(UserWarning, match="`binsize` and `nbins` are deprecated"):
        cc.scatter(model="SW_2", nbins=5, reg_method="odr")
    cc.scatter(model="SW_2", title="t", xlabel="x", ylabel="y")
    cc.scatter(model="SW_2", show_points=True)
    cc.scatter(model="SW_2", show_points=100)
    cc.scatter(model="SW_2", show_points=0.75)
    cc.scatter(model="SW_2", show_density=True)
    cc.scatter(model="SW_2", show_points=0.75, show_density=True)
    cc.scatter(model="SW_2", observation="HKNA", skill_table=True)
    # cc.scatter(model="SW_2", binsize=0.5, backend="plotly")
    assert True
    plt.close("all")


def test_mm_taylor(cc):
    cc.taylor(model="SW_1", observation=[0, 1])
    cc.taylor(normalize_std=True)
    cc.taylor(figsize=(4, 4))
    cc.taylor(model="SW_2", start="2017-10-28")
    cc[0].taylor(model=0, end="2017-10-29")
    assert True
    plt.close("all")


def test_mm_plot_timeseries(cc):
    cc["EPL"].plot_timeseries()
    cc["EPL"].plot_timeseries(title="t", figsize=(3, 3))

    # cc["EPL"].plot_timeseries(backend="plotly")
    with pytest.raises(ValueError):
        cc["EPL"].plot_timeseries(backend="mpl")
    plt.close("all")
