from datetime import timedelta
import numpy as np
import pandas as pd
import pytest
import mikeio
import modelskill as ms
from modelskill import ModelResult
import modelskill
from modelskill.observation import PointObservation, TrackObservation


@pytest.fixture
def o1():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def o2():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def o2_gaps():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    obs = mikeio.read(fn, items=0).to_dataframe().rename(columns=dict(Hm0="obs")) + 1
    dt = pd.Timedelta(180, unit="s")
    obs.index = obs.index - dt
    obs.index = obs.index.round("S")
    return PointObservation(obs, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def o3():
    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    return TrackObservation(fn, item=3, name="c2")


@pytest.fixture
def mr12_gaps():
    fn = "tests/testdata/SW/ts_storm_4.dfs0"
    df1 = mikeio.read(fn, items=0).to_dataframe()
    df1 = df1.resample("2H").nearest()
    df1 = df1.rename(columns={df1.columns[0]: "mr1"})
    df2 = df1.copy().rename(columns=dict(mr1="mr2")) - 1

    # keep 2017-10-28 00:00 and 2017-10-29 00:00
    # but remove the 11 steps in between
    df2.loc["2017-10-28 01:00":"2017-10-28 23:00"] = np.nan
    mr1 = ModelResult(df1, name="mr1")
    mr2 = ModelResult(df2, name="mr2")
    return mr1, mr2


@pytest.fixture
def mr1():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    return ModelResult(fn, item=0, name="SW_1")


@pytest.fixture
def mr2():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    return ModelResult(fn, item=0, name="SW_2")


@pytest.fixture
def mr3():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v3.dfsu"
    return ModelResult(fn, item=0, name="SW_3")


def test_compare_multi_obs_multi_model(o1, o2, o3, mr1, mr2):
    cc = ms.compare([o1, o2, o3], [mr1, mr2])
    assert cc.n_models == 2
    assert cc.n_observations == 3
    assert cc["c2"].n_points == 113
    assert cc["HKNA"].name == "HKNA"


def test_compare_dataarray(o1, o3):
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    da = mikeio.read(fn, time=slice("2017-10-28 00:00", None))[0]  # Skip warm-up period

    # Using a mikeio.DataArray instead of a Dfs file, makes it possible to select a subset of data

    cc = ms.compare([o1, o3], da)
    assert cc.n_models == 1
    assert cc["c2"].n_points == 41

    da2 = mikeio.read(fn, area=[0, 2, 52, 54], time=slice("2017-10-28 00:00", None))[
        0
    ]  # Spatio/temporal subset

    cc2 = ms.compare([o1, o3], da2)
    assert cc2["c2"].n_points == 19


def test_extract_gaps1(o1, mr3):
    # obs has 278 steps (2017-10-27 18:00 to 2017-10-29 18:00) (10min data with gaps)
    # model SW_3 has 5 timesteps:
    # 2017-10-27 18:00:00  1.880594
    # 2017-10-27 21:00:00  1.781904
    # 2017-10-28 00:00:00  1.819505   (not in obs)
    # 2017-10-28 03:00:00  2.119306
    # 2017-10-29 18:00:00  3.249600

    cmp = ms.compare(o1, mr3)
    assert cmp.n_points == 278

    # accept only 1 hour gaps (even though the model has 3 hour timesteps)
    # expect only exact matches (4 of 5 model timesteps are in obs)
    cmp = ms.compare(o1, mr3, max_model_gap=3600)
    assert cmp.n_points == 4

    # accept only 3 hour gaps
    # should disregard everything after 2017-10-28 03:00
    # except a single point 2017-10-29 18:00 (which is hit spot on)
    cmp = ms.compare(o1, mr3, max_model_gap=10800)
    assert cmp.n_points == 48 + 1

    # accept gaps up to 2 days (all points should be included)
    cmp = ms.compare(o1, mr3, max_model_gap=2 * 24 * 60 * 60)
    assert cmp.n_points == 278


def test_extract_gaps2(o2_gaps, mr12_gaps):
    # mr2 has no data between 2017-10-28 00:00 and 2017-10-29 00:00
    # we therefore expect the the 24 observations in this interval to be removed
    mr1, mr2 = mr12_gaps
    # con1 = Connector(o2_gaps, mr1)
    # con2 = Connector(o2_gaps, mr2)
    # con12 = Connector(o2_gaps, [mr1, mr2])

    cmp = ms.compare(o2_gaps, [mr1, mr2])  # con12.extract()  # no max gap argument
    assert cmp.data["mr1"].count() == 66
    assert cmp.data["mr2"].count() == 66

    # no gap in mr1
    cmp = ms.compare(o2_gaps, mr1, max_model_gap=7200)
    assert cmp.data["mr1"].count() == 66

    # one day gap in mr2
    cmp = ms.compare(o2_gaps, mr2, max_model_gap=7200)
    assert cmp.data["mr2"].count() == 42  # 66 - 24
    assert cmp.data["mr2"].sel(time="2017-10-28").count() == 0

    # will syncronize the two models,
    # so gap in one will remove points from the other
    cmp = ms.compare(o2_gaps, [mr1, mr2], max_model_gap=7200)
    assert cmp.data["mr1"].count() == 42
    assert cmp.data["mr2"].count() == 42

    # the 24 hour gap (86400 seconds) in the file cannot be filled
    # with the max_model_gap=27200
    cmp = ms.compare(o2_gaps, mr2, max_model_gap=27200)
    assert cmp.data["mr2"].count() == 42
    assert cmp.data["mr2"].sel(time="2017-10-28").count() == 0


def test_extract_gaps_big(o2_gaps, mr12_gaps):
    _, mr2 = mr12_gaps
    cmp = ms.compare(o2_gaps, mr2, max_model_gap=86401)  # 24 hours + 1 second
    assert cmp.data["mr2"].count() == 66  # no data removed


def test_extract_gaps_small(o2_gaps, mr12_gaps):
    _, mr2 = mr12_gaps
    # with pytest.warns(UserWarning, match="No overlapping data"):
    cmp = ms.compare(o2_gaps, mr2, max_model_gap=10)  # no data with that small gap
    assert cmp.n_points == 0


def test_extract_gaps_negative(o2_gaps, mr12_gaps):
    _, mr2 = mr12_gaps
    # with pytest.warns(UserWarning, match="No overlapping data"):
    cmp = ms.compare(o2_gaps, mr2, max_model_gap=-10)
    assert cmp.n_points == 0


def test_compare_gaps_types(o2_gaps, mr12_gaps):
    mr1, mr2 = mr12_gaps

    gap_seconds = 7200
    gaps = [
        pd.Timedelta(gap_seconds, unit="s"),
        np.timedelta64(gap_seconds, "s"),
        timedelta(seconds=gap_seconds),
    ]
    for gap in gaps:
        cmp = ms.compare(o2_gaps, [mr1, mr2], max_model_gap=gap)
        assert cmp.data["mr1"].count() == 42


def test_small_multi_model_shifted_time_compare():
    obs = pd.DataFrame(
        {"HKNA": [1.1, 2.0, 3.0, 4.0]}, index=pd.date_range("2017-01-01", periods=4)
    )
    mod = pd.DataFrame(
        {"Simple": [1.1, 2.0, 3.0]}, index=pd.date_range("2017-01-01", periods=3)
    )

    mod2 = pd.DataFrame(
        {"NotSimple": [2.1, 3.0, 4.0]}, index=pd.date_range("2017-01-02", periods=3)
    )

    # observation has four timesteps, but only three of them are in the Simple model and three in the NotSimple model
    # the number of overlapping points for all three datasets are 2, but three if we look at the models individually

    cmp1 = ms.compare(obs=obs, mod=mod)
    assert cmp1.n_points == 3

    cmp2 = ms.compare(obs=obs, mod=mod2)
    assert cmp2.n_points == 3

    mcmp = ms.compare(obs=obs, mod=[mod, mod2])
    assert mcmp.n_points == 2


def test_matched_data_single_model():
    df = pd.DataFrame(
        {"ts_1": [1.1, 2.0, 3.0, 4.0], "sensor_a": [0.9, 2.0, 3.0, 4.1]},
        index=pd.date_range("2017-01-01", periods=4),
    )

    cmp = ms.from_matched(df, obs_item="sensor_a")
    assert cmp.n_points == 4


def test_matched_data_quantity():

    df = pd.DataFrame(
        {"ts_1": [1.1, 2.0, 3.0, 4.0], "sensor_a": [0.9, 2.0, 3.0, 4.1]},
        index=pd.date_range("2017-01-01", periods=4),
    )
    quantity = modelskill.Quantity(name="Water level", unit="m")
    cmp = ms.from_matched(df, obs_item="sensor_a", quantity=quantity)

    # Model and observation have the same quantity by definition
    assert cmp.quantity == quantity


def test_matched_data_multiple_models():
    df = pd.DataFrame(
        {
            "cal_42": [1.1, 2.0, 3.0, 4.0],
            "cal_43": [0.9, 2.0, 3.0, 4.01],
            "sensor_a": [0.9, 2.0, 3.0, 4.1],
        },
        index=pd.date_range("2017-01-01", periods=4),
    )

    # TODO not sure about the _item suffix
    cmp = ms.from_matched(df, obs_item="sensor_a")
    assert cmp.n_points == 4
    assert cmp.n_models == 2


def test_matched_data_multiple_models_additional_columns():
    df = pd.DataFrame(
        {
            "cal_42": [1.1, 2.0, 3.0, 4.0],
            "cal_43": [0.9, 2.0, 3.0, 4.01],
            "sensor_a": [0.9, 2.0, 3.0, 4.1],
            "additional": [0.9, 2.0, 3.0, 4.1],
        },
        index=pd.date_range("2017-01-01", periods=4),
    )

    # TODO not sure about the _item suffix
    cmp = ms.from_matched(df, obs_item="sensor_a", mod_items=["cal_42", "cal_43"])
    assert cmp.n_points == 4
    assert cmp.n_models == 2
