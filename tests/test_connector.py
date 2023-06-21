from datetime import timedelta
import numpy as np
import pytest
import pandas as pd

import mikeio

from modelskill import ModelResult
from modelskill import PointObservation, TrackObservation
from modelskill import Connector
from modelskill.connection import PointConnector


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
def con13(o1, mr3):
    return Connector(o1, mr3)


@pytest.fixture
def con31(o1, o2, o3, mr1):
    return Connector([o1, o2, o3], mr1)


@pytest.fixture
def con32(o1, o2, o3, mr1, mr2):
    return Connector([o1, o2, o3], [mr1, mr2])


def test_point_connector_repr(o1, mr1):
    con = PointConnector(o1, mr1)
    txt = repr(con)
    assert "PointConnector" in txt


def test_connector_add(o1, mr1):
    con = Connector()
    con.add(o1, mr1, validate=False)
    assert len(con.observations) == 1


# TODO: remove, obsolete (has been moved to test_compare.py)
def test_connector_dataarray(o1, o3):

    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    da = mikeio.read(fn, time=slice("2017-10-28 00:00", None))[0]  # Skip warm-up period

    # Using a mikeio.DataArray instead of a Dfs file, makes it possible to select a subset of data

    c = Connector([o1, o3], da)
    assert c.n_models == 1

    cc = c.extract()
    assert cc.n_models == 1
    assert cc["c2"].n_points == 41

    da2 = mikeio.read(fn, area=[0, 2, 52, 54], time=slice("2017-10-28 00:00", None))[
        0
    ]  # Spatio/temporal subset

    c2 = Connector([o1, o3], da2)
    cc2 = c2.extract()
    assert cc2["c2"].n_points == 19


def test_connector_add_two_models(
    o1: PointObservation, mr1: ModelResult, mr2: ModelResult
):

    con = Connector(o1, [mr1, mr2])

    assert con.n_models == 2
    cc = con.extract()
    assert cc.n_models == 2

    # Alternative specification using .add() should be identical
    con2 = Connector()
    con2.add(o1, mr1)
    con2.add(o1, mr2)

    assert con2.n_models == 2
    cc2 = con2.extract()
    assert cc2.n_models == 2


def test_connector_add_two_model_dataframes(
    o1: PointObservation, mr1: ModelResult, mr2: ModelResult
):
    mr1_extr = mr1.extract(o1)
    # mr1_df = mr1._extract_point_dfsu(x=o1.x, y=o1.y, item=0).to_dataframe()
    mr2_extr = mr2.extract(o1)
    # mr2_df = mr2._extract_point_dfsu(x=o1.x, y=o1.y, item=0).to_dataframe()

    assert isinstance(mr1_extr.data, pd.DataFrame)
    assert isinstance(mr2_extr.data, pd.DataFrame)

    assert len(mr1_extr.data.columns == 1)
    assert len(mr2_extr.data.columns == 1)

    assert len(mr1_extr.data) > 1  # Number of rows
    assert len(mr2_extr.data) > 1  # Number of rows

    con = Connector(o1, [mr1_extr, mr2_extr])

    assert con.n_models == 2
    cc = con.extract()
    assert cc.n_models == 2

    # Alternative specification using .add() should be identical
    con2 = Connector()
    con2.add(o1, mr1_extr)
    con2.add(o1, mr2_extr)

    assert con2.n_models == 2
    cc2 = con2.extract()
    assert cc2.n_models == 2


# def test_add_observation_eum_validation(hd_oresund_2d, klagshamn):
#     mr = ModelResult(hd_oresund_2d)
#     with pytest.raises(ValueError):
#         # EUM type doesn't match
#         mr.add_observation(klagshamn, item=0)

#     klagshamn.itemInfo = mikeio.ItemInfo(mikeio.EUMType.Surface_Elevation)
#     mr = ModelResult(hd_oresund_2d)
#     mr.add_observation(klagshamn, item=0)
#     assert len(mr.observations) == 1

#     klagshamn.itemInfo = mikeio.ItemInfo(
#         mikeio.EUMType.Surface_Elevation, unit=mikeio.EUMUnit.feet
#     )
#     with pytest.raises(ValueError):
#         # EUM unit doesn't match
#         mr.add_observation(klagshamn, item=0)


# TODO: remove, obsolete (has been moved to test_compare.py)
def test_extract(con32):
    collection = con32.extract()
    collection["HKNA"].name == "HKNA"


def test_plot_positions(con32):
    con32.plot_observation_positions()


def test_plot_data_coverage(con31):
    con31.plot_temporal_coverage()


# TODO: remove, obsolete
def test_extract_gaps1(con13):
    # obs has 278 steps (2017-10-27 18:00 to 2017-10-29 18:00) (10min data with gaps)
    # model SW_3 has 5 timesteps:
    # 2017-10-27 18:00:00  1.880594
    # 2017-10-27 21:00:00  1.781904
    # 2017-10-28 00:00:00  1.819505   (not in obs)
    # 2017-10-28 03:00:00  2.119306
    # 2017-10-29 18:00:00  3.249600

    cc = con13.extract()
    assert cc.n_points == 278

    # accept only 1 hour gaps (even though the model has 3 hour timesteps)
    # expect only exact matches (4 of 5 model timesteps are in obs)
    cc = con13.extract(max_model_gap=3600)
    assert cc.n_points == 4

    # accept only 3 hour gaps
    # should disregard everything after 2017-10-28 03:00
    # except a single point 2017-10-29 18:00 (which is hit spot on)
    cc = con13.extract(max_model_gap=10800)
    assert cc.n_points == 48 + 1

    # accept gaps up to 2 days (all points should be included)
    cc = con13.extract(max_model_gap=2 * 24 * 60 * 60)
    assert cc.n_points == 278


# TODO: remove, obsolete
def test_extract_gaps2(o2_gaps, mr12_gaps):

    # mr2 has no data between 2017-10-28 00:00 and 2017-10-29 00:00
    # we therefore expect the the 24 observations in this interval to be removed
    mr1, mr2 = mr12_gaps
    con1 = Connector(o2_gaps, mr1)
    con2 = Connector(o2_gaps, mr2)
    con12 = Connector(o2_gaps, [mr1, mr2])

    cc = con12.extract()  # no max gap argument
    assert cc[0].data["mr1"].count() == 66
    assert cc[0].data["mr2"].count() == 66

    # no gap in mr1
    cc = con1.extract(max_model_gap=7200)
    assert cc[0].data["mr1"].count() == 66

    # one day gap in mr2
    cc = con2.extract(max_model_gap=7200)
    assert cc[0].data["mr2"].count() == 42  # 66 - 24
    assert cc[0].data["mr2"].sel(time="2017-10-28").count() == 0

    # will syncronize the two models,
    # so gap in one will remove points from the other
    cc = con12.extract(max_model_gap=7200)
    assert cc[0].data["mr1"].count() == 42
    assert cc[0].data["mr2"].count() == 42

    # the 24 hour gap (86400 seconds) in the file cannot be filled
    # with the max_model_gap=27200
    cc = con2.extract(max_model_gap=27200)
    assert cc[0].data["mr2"].count() == 42
    assert cc[0].data["mr2"].sel(time="2017-10-28").count() == 0


# TODO: remove, obsolete
def test_extract_gaps_big(o2_gaps, mr12_gaps):
    _, mr2 = mr12_gaps
    con2 = Connector(o2_gaps, mr2)
    cc = con2.extract(max_model_gap=86401)  # 24 hours + 1 second
    assert cc[0].data["mr2"].count() == 66  # no data removed


# TODO: remove, obsolete
def test_extract_gaps_small(o2_gaps, mr12_gaps):
    _, mr2 = mr12_gaps
    con2 = Connector(o2_gaps, mr2)
    with pytest.warns(UserWarning, match="No overlapping data"):
        cc = con2.extract(max_model_gap=10)  # no data with that small gap
    assert cc.n_comparers == 0


# TODO: remove, obsolete
def test_extract_gaps_negative(o2_gaps, mr12_gaps):
    _, mr2 = mr12_gaps
    con2 = Connector(o2_gaps, mr2)
    with pytest.warns(UserWarning, match="No overlapping data"):
        cc = con2.extract(max_model_gap=-10)
    assert cc.n_comparers == 0


# TODO: remove, obsolete
def test_extract_gaps_types(o2_gaps, mr12_gaps):
    mr1, mr2 = mr12_gaps
    con2 = Connector(o2_gaps, [mr1, mr2])

    gap_seconds = 7200
    gaps = [
        pd.Timedelta(gap_seconds, unit="s"),
        np.timedelta64(gap_seconds, "s"),
        timedelta(seconds=gap_seconds),
    ]
    for gap in gaps:
        cc = con2.extract(max_model_gap=gap)
        assert cc[0].data["mr1"].count() == 42
