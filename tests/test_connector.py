import pytest
import pandas as pd

import mikeio

from fmskill import ModelResult
from fmskill import PointObservation, TrackObservation
from fmskill import Connector
from fmskill.connection import PointConnector


@pytest.fixture
def mr1():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    return ModelResult(fn, name="SW_1")


@pytest.fixture
def mr2():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    return ModelResult(fn, name="SW_2")


@pytest.fixture
def mr3():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v3.dfsu"
    return ModelResult(fn, name="SW_3")


@pytest.fixture
def mr4():
    fn = "tests/testdata/SW/HKNA_Hm0_Model.dfs0"
    return ModelResult(fn, name="SW_4")


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
def con11(o1, mr1):
    return Connector(o1, mr1[0])


@pytest.fixture
def con11_b(o1, mr3):
    return Connector([o1], mr3[0])


@pytest.fixture
def con11_c(o1, mr4):
    return Connector([o1], mr4)


@pytest.fixture
def con31(o1, o2, o3, mr1):
    return Connector([o1, o2, o3], mr1[0])


@pytest.fixture
def con32(o1, o2, o3, mr1, mr2):
    return Connector([o1, o2, o3], [mr1[0], mr2[0]])


@pytest.fixture
def con33(o1, mr3):
    return Connector([o1], mr3[0])


def test_point_connector_repr(o1, mr1):
    con = PointConnector(o1, mr1[0])
    txt = repr(con)
    assert "PointConnector" in txt


def test_connector_add(o1, mr1):
    con = Connector()
    con.add(o1, mr1[0], validate=False)
    assert len(con.observations) == 1


def test_connector_add_two_models(
    o1: PointObservation, mr1: ModelResult, mr2: ModelResult
):

    con = Connector(o1, [mr1[0], mr2[0]])

    assert con.n_models == 2
    cc = con.extract()
    assert cc.n_models == 2

    # Alternative specification using .add() should be identical
    con2 = Connector()
    con2.add(o1, mr1[0])
    con2.add(o1, mr2[0])

    assert con2.n_models == 2
    cc2 = con2.extract()
    assert cc2.n_models == 2


def test_connector_add_two_model_dataframes(
    o1: PointObservation, mr1: ModelResult, mr2: ModelResult
):

    mr1_df = mr1[0]._extract_point_dfsu(x=o1.x, y=o1.y, item=0).to_dataframe()
    mr2_df = mr2[0]._extract_point_dfsu(x=o1.x, y=o1.y, item=0).to_dataframe()

    assert isinstance(mr1_df, pd.DataFrame)
    assert isinstance(mr2_df, pd.DataFrame)

    assert len(mr1_df.columns == 1)
    assert len(mr2_df.columns == 1)

    assert len(mr1_df) > 1  # Number of rows
    assert len(mr2_df) > 1  # Number of rows

    con = Connector(o1, [mr1_df, mr2_df])

    assert con.n_models == 2
    cc = con.extract()
    assert cc.n_models == 2

    # Alternative specification using .add() should be identical
    con2 = Connector()
    con2.add(o1, mr1_df)
    con2.add(o1, mr2_df)

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


def test_add_fail(o2, mr1):
    # mr.add_observation(Hm0_EPL)  # infer item by EUM
    # assert len(mr.observations) == 1

    con = Connector()
    with pytest.raises(Exception):
        with pytest.warns(UserWarning):
            # item not specified
            con.add(o2, mr1)

    eumHm0 = mikeio.EUMType.Significant_wave_height
    o2.itemInfo = mikeio.ItemInfo(eumHm0, unit=mikeio.EUMUnit.feet)
    with pytest.raises(Exception):
        with pytest.warns(UserWarning):
            # EUM unit doesn't match
            con.add(o2, mr1[0])

    o2.itemInfo = mikeio.ItemInfo(mikeio.EUMType.Water_Level, unit=mikeio.EUMUnit.meter)
    with pytest.raises(Exception):
        with pytest.warns(UserWarning):
            # EUM type doesn't match
            con.add(o2, mr1[0])


def test_extract(con32):
    collection = con32.extract()
    collection["HKNA"].name == "HKNA"


def test_plot_positions(con32):
    con32.plot_observation_positions()


def test_plot_data_coverage(con31):
    con31.plot_temporal_coverage()


# def test_extract_gaps(con33):
#     collection = con33.extract()
#     assert collection.n_points==28
