import numpy as np
import pytest

import mikeio
from modelskill import ModelResult
from modelskill.connection import Connector
from modelskill.model import DfsuModelResult, PointModelResult, TrackModelResult
from modelskill.observation import PointObservation, TrackObservation


@pytest.fixture
def hd_oresund_2d():
    return "tests/testdata/Oresund2D.dfsu"


# TODO: replace with shorter dfs0
@pytest.fixture
def klagshamn():
    fn = "tests/testdata/smhi_2095_klagshamn.dfs0"
    return PointObservation(fn, item=0, x=366844, y=6154291, name="Klagshamn")


@pytest.fixture
def drogden():

    # >>> from pyproj import Transformer
    # >>> t = Transformer.from_crs(4326,32633, always_xy=True)
    # >>> t.transform(12.7113,55.5364)
    # (355568.6130331255, 6156863.0187071245)

    fn = "tests/testdata/dmi_30357_Drogden_Fyr.dfs0"
    return PointObservation(fn, item=0, x=355568.0, y=6156863.0)


@pytest.fixture
def sw_dutch_coast():
    return "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"


@pytest.fixture
def sw_total_windsea():
    return "tests/testdata/SW/SW_Tot_Wind_Swell.dfsu"


@pytest.fixture
def Hm0_HKNA():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def wind_HKNA():
    fn = "tests/testdata/SW/HKNA_wind.dfs0"
    return PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def Hm0_EPL():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def Hm0_C2():
    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    return TrackObservation(fn, item=3, name="C2")


def test_dfsu_repr(hd_oresund_2d):
    mr = ModelResult(hd_oresund_2d, name="Oresund2D", item="Surface elevation")
    txt = repr(mr)
    assert "Oresund2D" in txt


def test_dfsu_properties(hd_oresund_2d):
    mr = ModelResult(hd_oresund_2d, name="Oresund2d", item="Surface elevation")

    assert mr.data.is_2d

    # Note != name of item
    assert mr.quantity.name == "Surface Elevation"

    # this is the unit, shortening it is a presentation concern
    assert mr.quantity.unit == "meter"


def test_dfsu_sw(sw_dutch_coast):
    mr = ModelResult(sw_dutch_coast, name="SW", item=0)

    assert isinstance(mr, DfsuModelResult)


# def test_model_dfsu(hd_oresund_2d):
#     mr = DfsuModelResult(hd_oresund_2d, item=0, "Oresund")
#     assert mr.n_items == 7
#     assert isinstance(mr, DfsModelResult)

#     mr0 = mr[0]
#     assert isinstance(mr0, DfsModelResultItem)
#     assert mr.item_names[0] == mr0.item_name

#     mr1 = mr["Surface elevation"]
#     assert mr.item_names[0] == mr1.item_name
#     assert mr.filename == mr1.filename
#     assert mr.name == mr1.name


def test_dfsu_dataarray(hd_oresund_2d):
    ds = mikeio.read(hd_oresund_2d)
    assert ds.n_items == 7
    da = ds[0]
    assert isinstance(da, mikeio.DataArray)

    mr = ModelResult(da, name="Oresund")
    assert mr.name == "Oresund"
    assert isinstance(mr.data, mikeio.DataArray)

    mr.name = "Oresund2"
    assert mr.name == "Oresund2"


def test_dfsu_factory(hd_oresund_2d):
    mr1 = ModelResult(hd_oresund_2d, name="myname", item=-1)
    assert isinstance(mr1, DfsuModelResult)
    assert mr1.name == "myname"

    mr2 = ModelResult(hd_oresund_2d, name="Oresund2d", item="Surface elevation")
    assert isinstance(mr2, DfsuModelResult)
    assert mr2.name == "Oresund2d"


# def test_extract_observation(sw_dutch_coast, Hm0_HKNA):
#     mr = ModelResult(sw_dutch_coast)
#     c = mr.extract_observation(Hm0_HKNA)  # infer item by EUM
#     assert c.n_points == 386


# def test_extract_observation_no_matching_item(sw_total_windsea, wind_HKNA):
#     mr = ModelResult(sw_total_windsea)  # No wind speed here !

#     with pytest.raises(Exception):  # More specific error?
#         _ = mr.extract_observation(wind_HKNA)


# TODO: move this test to test_connector.py
def test_extract_observation_total_windsea_swell_not_possible(
    sw_total_windsea, Hm0_HKNA
):
    mr = ModelResult(sw_total_windsea, name="SW", item="Sign. Wave Height, S")
    """
    Items:
        0:  Sign. Wave Height <Significant wave height> (meter)
        1:  Sign. Wave Height, W <Significant wave height> (meter)
        2:  Sign. Wave Height, S <Significant wave height> (meter)
    """

    # with pytest.raises(Exception):
    #     c = mr.extract_observation(Hm0_HKNA)  # infer item by EUM is ambigous

    # Specify Swell item explicitely
    c = Connector(Hm0_HKNA, mr).extract()
    assert c.n_points > 0


# TODO: move this test to test_connector.py
def test_extract_observation_validation(hd_oresund_2d, klagshamn):
    mr = ModelResult(hd_oresund_2d, item=0)
    with pytest.raises(Exception):
        c = Connector(klagshamn, mr, validate=True).extract()

    # No error if validate==False
    c = Connector(klagshamn, mr, validate=False).extract()
    assert c.n_points > 0


# TODO: move this test to test_connector.py
def test_extract_observation_outside(hd_oresund_2d, klagshamn):
    mr = ModelResult(hd_oresund_2d, item=0)
    # correct eum, but outside domain
    klagshamn.y = -10
    with pytest.raises(ValueError):
        _ = Connector(klagshamn, mr, validate=True).extract()

        # _ = mr.extract_observation(klagshamn, validate=True)


def test_dfsu_extract_point(sw_dutch_coast, Hm0_EPL):
    mr1 = ModelResult(sw_dutch_coast, item=0, name="SW1")
    mr_extr_1 = mr1.extract(Hm0_EPL.copy())
    # df1 = mr1._extract_point(Hm0_EPL)
    assert mr_extr_1.data.columns == ["SW1"]
    assert len(mr_extr_1.data) == 23

    da = mikeio.read(sw_dutch_coast)[0]
    mr2 = ModelResult(da, name="SW1")
    mr_extr_2 = mr2.extract(Hm0_EPL.copy())

    assert mr_extr_1.data.columns == mr_extr_2.data.columns
    assert np.all(mr_extr_1.data == mr_extr_2.data)

    c1 = mr1.extract(Hm0_EPL.copy())
    c2 = mr2.extract(Hm0_EPL.copy())
    assert isinstance(c1, PointModelResult)
    assert isinstance(c2, PointModelResult)
    assert np.all(c1.data == c2.data)
    # c1.observation.itemInfo == Hm0_EPL.itemInfo
    # assert len(c1.observation.data.index.difference(Hm0_EPL.data.index)) == 0


def test_dfsu_extract_track(sw_dutch_coast, Hm0_C2):
    mr1 = ModelResult(sw_dutch_coast, item=0, name="SW1")
    mr_track1 = mr1.extract(Hm0_C2)
    df1 = mr_track1.data
    assert list(df1.columns) == ["x", "y", "SW1"]
    assert len(df1) == 113

    da = mikeio.read(sw_dutch_coast)[0]
    mr2 = ModelResult(da, name="SW1")
    mr_track2 = mr2.extract(Hm0_C2.copy())
    df2 = mr_track2.data

    assert list(df1.columns) == list(df2.columns)
    assert np.all(df1 == df2)

    c1 = mr1.extract(Hm0_C2.copy())
    c2 = mr2.extract(Hm0_C2.copy())
    assert isinstance(c1, TrackModelResult)
    assert isinstance(c2, TrackModelResult)
    assert np.all(c1.data == c2.data)
    # c1.observation.itemInfo == Hm0_C2.itemInfo
    # assert len(c1.observation.data.index.difference(Hm0_C2.data.index)) == 0
