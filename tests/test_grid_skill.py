import pytest
import pandas as pd

import modelskill as ms


@pytest.fixture
def cmp1() -> ms.Comparer:
    fn = "tests/testdata/NorthSeaHD_and_windspeed.dfsu"
    mr = ms.model_result(fn, item=0, name="HD")
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    df = pd.read_csv(fn, index_col=0, parse_dates=True)
    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        o1 = ms.TrackObservation(df, item=2, name="alti")

    return ms.match(o1, mr)


@pytest.fixture
def o1() -> ms.PointObservation:
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return ms.PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def o2() -> ms.PointObservation:
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return ms.PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def o3() -> ms.TrackObservation:
    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    return ms.TrackObservation(fn, item=3, name="c2")


@pytest.fixture
def cc2(o1, o2, o3) -> ms.ComparerCollection:
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    mr1 = ms.model_result(fn, item=0, name="SW_1")
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    mr2 = ms.model_result(fn, item=0, name="SW_2")

    return ms.match([o1, o2, o3], [mr1, mr2])


def test_gridded_skill_multi_model(cc2) -> None:
    gs = cc2.gridded_skill(bins=3, metrics=["rmse", "bias"])
    assert len(gs.x) == 3
    assert len(gs.y) == 3
    assert len(gs.mod_names) == 2
    assert len(gs.obs_names) == 3
    assert len(gs.metrics) == 3


def test_gridded_skill_sel_model(cc2) -> None:
    gs = cc2.gridded_skill(bins=3, metrics=["rmse", "bias"])
    gs2 = gs.sel(model="SW_1")
    gs2.rmse.plot()

    with pytest.raises(KeyError):
        gs.sel(model="bad_model")


@pytest.mark.skipif(pd.__version__ < "2.0.0", reason="requires newer pandas")
def test_gridded_skill_is_subsettable(cc2) -> None:
    gs = cc2.gridded_skill(bins=3, metrics=["rmse", "bias"])
    gs.data.rmse.sel(x=2, y=53.5, method="nearest").values == pytest.approx(0.10411702)

    cmp = cc2[0]

    _ = cmp.gridded_skill(bins=3, metrics=["rmse", "bias"])

    # TODO check if this makes sense
    # gs2.data.rmse.sel(x=2, y=53.5, method="nearest").values == pytest.approx(0.10411702)


def test_gridded_skill_plot(cmp1) -> None:
    gs = cmp1.gridded_skill(metrics=["rmse", "bias"])
    gs.bias.plot()


def test_gridded_skill_plot_multi_model(cc2) -> None:
    gs = cc2.gridded_skill(by=["model"], metrics=["rmse", "bias"])
    gs["bias"].plot()


def test_gridded_skill_plot_multi_model_fails(cc2) -> None:
    gs = cc2.gridded_skill(by=["model"], metrics=["rmse", "bias"])
    with pytest.raises(KeyError):
        gs["bad_metric"]
