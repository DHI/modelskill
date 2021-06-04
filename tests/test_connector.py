from mikeio import eum
import pytest

from fmskill import ModelResult
from fmskill import PointObservation, TrackObservation
from fmskill import Connector
from fmskill.connection import PointConnector, TrackConnector


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
def con11(o1, mr1):
    return Connector(o1, mr1[0])


@pytest.fixture
def con31(o1, o2, o3, mr1):
    return Connector([o1, o2, o3], mr1[0])


@pytest.fixture
def con32(o1, o2, o3, mr1, mr2):
    return Connector([o1, o2, o3], [mr1[0], mr2[0]])


def test_point_connector_repr(o1, mr1):
    con = PointConnector(o1, mr1[0])
    txt = repr(con)
    assert "PointConnector" in txt


def test_connector_add(o1, mr1):
    con = Connector()
    con.add(o1, mr1[0], validate=False)
    assert len(con.observations) == 1


# def test_add_observation_eum_validation(hd_oresund_2d, klagshamn):
#     mr = ModelResult(hd_oresund_2d)
#     with pytest.raises(ValueError):
#         # EUM type doesn't match
#         mr.add_observation(klagshamn, item=0)

#     klagshamn.itemInfo = eum.ItemInfo(eum.EUMType.Surface_Elevation)
#     mr = ModelResult(hd_oresund_2d)
#     mr.add_observation(klagshamn, item=0)
#     assert len(mr.observations) == 1

#     klagshamn.itemInfo = eum.ItemInfo(
#         eum.EUMType.Surface_Elevation, unit=eum.EUMUnit.feet
#     )
#     with pytest.raises(ValueError):
#         # EUM unit doesn't match
#         mr.add_observation(klagshamn, item=0)


def test_add_fail(o2, mr1):
    # mr.add_observation(Hm0_EPL)  # infer item by EUM
    # assert len(mr.observations) == 1

    con = Connector()
    with pytest.raises(Exception):
        # item not specified
        con.add(o2, mr1)

    eumHm0 = eum.EUMType.Significant_wave_height
    o2.itemInfo = eum.ItemInfo(eumHm0, unit=eum.EUMUnit.feet)
    with pytest.raises(Exception):
        # EUM unit doesn't match
        con.add(o2, mr1[0])

    o2.itemInfo = eum.ItemInfo(eum.EUMType.Water_Level, unit=eum.EUMUnit.meter)
    with pytest.raises(Exception):
        # EUM type doesn't match
        con.add(o2, mr1[0])


def test_extract(con32):
    collection = con32.extract()
    collection["HKNA"].name == "HKNA"


def test_plot_positions(con32):
    con32.plot_observation_positions()


def test_plot_data_coverage(con31):
    con31.plot_temporal_coverage()
