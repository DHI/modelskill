import pytest
import numpy as np

from fmskill.model import ModelResult, ModelResultCollection
from fmskill.observation import PointObservation, TrackObservation
import fmskill.metrics as mtr


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
def mrc(mr1, mr2):
    return ModelResultCollection([mr1, mr2])


@pytest.fixture
def cc(mr1, mr2, o1, o2, o3):
    mrc = ModelResultCollection([mr1, mr2])
    mrc.add_observation(o1, item=0)
    mrc.add_observation(o2, item=0)
    mrc.add_observation(o3, item=0)
    return mrc.extract()


def test_mrc_repr(mrc):
    txt = repr(mrc)
    assert "ModelResultCollection" in txt


def test_add_observation(mrc, o1):
    mrc.add_observation(o1, item=0)
    assert len(mrc.observations) == 1


def test_extract(mrc, o1, o2, o3):
    mrc.add_observation(o1, item=0)
    mrc.add_observation(o2, item=0)
    mrc.add_observation(o3, item=0)
    cc = mrc.extract()
    assert cc.n_points > 0
    assert "ComparerCollection" in repr(cc)
    assert "PointComparer" in repr(cc["EPL"])
    assert "TrackComparer" in repr(cc[2])


def test_add_comparer(mr1, mr2, o1, o2):
    cc1 = mr1.add_observation(o1, item=0).extract()
    cc2 = mr2.add_observation(o2, item=0).extract()
    cc = cc1 + cc2
    assert cc.n_points > 0
    assert "ComparerCollection" in repr(cc)
    assert "PointComparer" in repr(cc["EPL"])
    assert "PointComparer" in repr(cc["HKNA"])


def test_add_same_comparer_twice(mr1, mr2, o1, o2):
    cc1 = mr1.add_observation(o1, item=0).extract()
    cc2 = mr2.add_observation(o2, item=0).extract()
    cc = cc1 + cc2
    assert len(cc) == 2
    cc = cc + cc2
    assert len(cc) == 2  # adding the same comparer again doesn't have any effect
    assert cc.n_points > 0
    assert "ComparerCollection" in repr(cc)
    assert "PointComparer" in repr(cc["EPL"])
    assert "PointComparer" in repr(cc["HKNA"])


def test_mm_skill(cc):
    df = cc.skill().df
    assert df.iloc[4].name[0] == "SW_2"
    assert df.iloc[4].name[1] == "HKNA"
    assert pytest.approx(df.iloc[4].mae, 1e-5) == 0.214476


def test_mm_skill_model(cc):
    df = cc.skill(model="SW_1").df
    assert df.loc["EPL"].n == 66
    assert df.loc["c2"].n == 113


def test_mm_skill_missing_model(cc):
    with pytest.raises(ValueError):
        cc.skill(model="SW_3")
    with pytest.raises(ValueError):
        cc.skill(model=999)
    with pytest.raises(ValueError):
        cc.skill(model=[999, "SW_2"])
    with pytest.raises(ValueError):
        cc.skill(model=[0.1])


def test_mm_skill_obs(cc):
    df = cc.skill(observation="c2").df
    assert len(df) == 2
    assert df.loc["SW_2"].bias == 0.08143105172057515

    df = cc.mean_skill(model=0, observation=[0, "c2"]).df
    assert df.si[0] == 0.10349949854443843


def test_mm_skill_missing_obs(cc, o1):
    with pytest.raises(KeyError):
        cc.skill(observation="imaginary_obs")
    with pytest.raises(IndexError):
        cc.skill(observation=999)
    with pytest.raises((KeyError, IndexError)):
        cc.skill(observation=["c2", 999])
    with pytest.raises(KeyError):
        cc.skill(observation=[o1])


def test_mm_skill_start_end(cc):
    df = cc.skill(model="SW_1", start="2017").df
    assert df.loc["EPL"].n == 66
    df = cc.skill(model="SW_1", end="2017-10-28 00:00:01").df
    assert df.loc["EPL"].n == 24
    df = cc.skill(model="SW_1", start="2017-10-28 00:00:01").df
    assert df.loc["EPL"].n == 42


def test_mm_skill_area(cc):
    bbox = [0.5, 52.5, 5, 54]
    df = cc.skill(model="SW_1", area=bbox).df
    assert pytest.approx(df.loc["HKNA"].urmse) == 0.29321445043385863
    bbox = np.array([0.5, 52.5, 5, 54])
    df = cc.skill(model="SW_1", area=bbox).df
    assert pytest.approx(df.loc["HKNA"].urmse) == 0.29321445043385863

    polygon = np.array([[6, 51], [0, 55], [0, 51], [6, 51]])
    df = cc.skill(model="SW_2", area=polygon).df
    assert "HKNA" not in df.index
    assert df.n[1] == 66
    assert pytest.approx(df.iloc[0].r2) == 0.9280893149478934

    # same as above but not closed
    polygon = np.array([[6, 51], [0, 55], [0, 51]])
    df = cc.skill(model="SW_2", area=polygon).df
    assert pytest.approx(df.iloc[0].r2) == 0.9932189179977318

    polygon = [6, 51, 0, 55, 0, 51, 6, 51]
    df = cc.skill(model="SW_2", area=polygon).df
    assert pytest.approx(df.iloc[0].r2) == 0.9932189179977318

    # same as above but not closed
    polygon = [6, 51, 0, 55, 0, 51]
    df = cc.skill(model="SW_2", area=polygon).df
    assert pytest.approx(df.iloc[0].r2) == 0.9932189179977318

    df = cc.mean_skill(area=polygon).df
    assert pytest.approx(df.loc["SW_2"].rmse) == 0.331661

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

    df = cc.skill(model="SW_1", metrics=[mtr.bias, "rmse"]).df
    assert df.loc["EPL"].bias == -0.07533533467221397
    assert df.loc["EPL"].rmse == 0.21635651988376833

    with pytest.raises(ValueError):
        cc.skill(model="SW_1", metrics=["mean_se"])
    with pytest.raises(AttributeError):
        cc.skill(model="SW_1", metrics=[mtr.fake])
    with pytest.raises(ValueError):
        cc.skill(model="SW_1", metrics=[47])


def test_mm_mean_skill(cc):
    df = cc.mean_skill()
    assert len(df) == 2
    df = cc.mean_skill(weights=[0.2, 0.3, 1.0])
    assert len(df) == 2
    df = cc.mean_skill(weights="points")
    assert len(df) == 2
    df = cc.mean_skill(weights=1)
    assert len(df) == 2
    df = cc.mean_skill(weights="equal")
    assert len(df) == 2
    with pytest.raises(ValueError):
        # too many weights
        cc.mean_skill(weights=[0.2, 0.3, 0.4, 0.5])


def test_mm_scatter(cc):
    cc.scatter(model="SW_1", observation=[0, 1])
    cc.scatter(model="SW_2", show_points=False)
    cc.scatter(model="SW_2", show_hist=False)
    cc.scatter(model="SW_2", binsize=0.5)
    cc.scatter(model="SW_2", nbins=5, reg_method="odr")
    cc.scatter(model="SW_2", title="t", xlabel="x", ylabel="y")
    # cc.scatter(model="SW_2", binsize=0.5, backend="plotly")
    assert True


def test_mm_plot_timeseries(cc):
    cc["EPL"].plot_timeseries()
    cc["EPL"].plot_timeseries(title="t", figsize=(3, 3))

    # cc["EPL"].plot_timeseries(backend="plotly")
    with pytest.raises(ValueError):
        cc["EPL"].plot_timeseries(backend="mpl")
