import pytest

from fmskill.model import ModelResult
from fmskill.observation import PointObservation
from mikeio import eum


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
def hd_oresund_2d():
    return "tests/testdata/Oresund2D.dfsu"


@pytest.fixture
def Hm0_HKNA():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def wind_HKNA():
    fn = "tests/testdata/SW/HKNA_Wind.dfs0"
    return PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def Hm0_EPL():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def sw_dutch_coast():
    return "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"


@pytest.fixture
def sw_total_windsea():
    return "tests/testdata/SW/SW_Tot_Wind_Swell.dfsu"


def test_dfs_object(hd_oresund_2d):
    mr = ModelResult(hd_oresund_2d)

    assert mr.dfs.is_2d


def test_ModelResultType(sw_dutch_coast):
    mr = ModelResult(sw_dutch_coast)

    assert mr.is_dfsu


def test_ModelResultType0():
    mr = ModelResult("tests/testdata/TS.dfs0")

    assert mr.is_dfs0


def test_add_observation(hd_oresund_2d, klagshamn):
    mr = ModelResult(hd_oresund_2d)
    mr.add_observation(klagshamn, item=0, validate_eum=False)
    assert len(mr.observations) == 1


def test_add_observation_eum_validation(hd_oresund_2d, klagshamn):
    mr = ModelResult(hd_oresund_2d)
    with pytest.raises(ValueError):
        # EUM type doesn't match
        mr.add_observation(klagshamn, item=0)

    klagshamn.itemInfo = eum.ItemInfo(eum.EUMType.Surface_Elevation)
    mr = ModelResult(hd_oresund_2d)
    mr.add_observation(klagshamn, item=0)
    assert len(mr.observations) == 1

    klagshamn.itemInfo = eum.ItemInfo(
        eum.EUMType.Surface_Elevation, unit=eum.EUMUnit.feet
    )
    with pytest.raises(ValueError):
        # EUM unit doesn't match
        mr.add_observation(klagshamn, item=0)


def test_add_observation_infer_item(sw_dutch_coast, Hm0_EPL):
    mr = ModelResult(sw_dutch_coast)
    mr.add_observation(Hm0_EPL)  # infer item by EUM
    assert len(mr.observations) == 1

    mr = ModelResult(sw_dutch_coast)
    Hm0_EPL.itemInfo = eum.ItemInfo(
        eum.EUMType.Significant_wave_height, unit=eum.EUMUnit.feet
    )
    with pytest.raises(Exception):
        # EUM unit doesn't match
        mr.add_observation(Hm0_EPL)


def test_extract(hd_oresund_2d, klagshamn, drogden):
    mr = ModelResult(hd_oresund_2d)

    mr.add_observation(klagshamn, item=0, validate_eum=False)
    mr.add_observation(drogden, item=0, validate_eum=False)
    collection = mr.extract()
    collection["Klagshamn"].name == "Klagshamn"


def test_extract_observation(sw_dutch_coast, Hm0_HKNA):
    mr = ModelResult(sw_dutch_coast)
    c = mr.extract_observation(Hm0_HKNA)  # infer item by EUM
    assert c.n_points == 386


def test_extract_observation_no_matching_item(sw_total_windsea, wind_HKNA):
    mr = ModelResult(sw_total_windsea)  # No wind speed here !

    with pytest.raises(Exception):  # More specific error?
        c = mr.extract_observation(wind_HKNA)


def test_extract_observation_total_windsea_swell_not_possible(
    sw_total_windsea, Hm0_HKNA
):
    mr = ModelResult(sw_total_windsea)
    """
    Items:
        0:  Sign. Wave Height <Significant wave height> (meter)
        1:  Sign. Wave Height, W <Significant wave height> (meter)
        2:  Sign. Wave Height, S <Significant wave height> (meter)
    """

    with pytest.raises(Exception):
        c = mr.extract_observation(Hm0_HKNA)  # infer item by EUM is ambigous

    c = mr.extract_observation(
        Hm0_HKNA, item="Sign. Wave Height, S"
    )  # Specify Swell item explicitely
    assert c.n_points > 0


def test_extract_observation_validation(hd_oresund_2d, klagshamn):
    mr = ModelResult(hd_oresund_2d)
    with pytest.raises(Exception):
        c = mr.extract_observation(klagshamn, item=0, validate=True)

    c = mr.extract_observation(klagshamn, item=0, validate=False)
    assert c.n_points > 0


def test_extract_observation_outside(hd_oresund_2d, klagshamn):
    mr = ModelResult(hd_oresund_2d)
    # correct eum, but outside domain
    klagshamn.itemInfo = eum.ItemInfo(eum.EUMType.Surface_Elevation)
    klagshamn.y = -10
    with pytest.raises(ValueError):
        c = mr.extract_observation(klagshamn, item=0, validate=True)


def test_plot_positions(sw_dutch_coast, Hm0_EPL, Hm0_HKNA):
    mr = ModelResult(sw_dutch_coast)
    mr.add_observation(Hm0_EPL, item=0)
    mr.add_observation(Hm0_HKNA, item=0)
    mr.plot_observation_positions()


def test_plot_data_coverage(sw_dutch_coast, Hm0_EPL, Hm0_HKNA):
    mr = ModelResult(sw_dutch_coast)
    mr.add_observation(Hm0_EPL, item=0)
    mr.add_observation(Hm0_HKNA, item=0)
    mr.plot_temporal_coverage()
