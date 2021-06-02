from datetime import datetime
import pytest

from fmskill.model import ModelResult, DataFrameModelResult, ModelResultInterface
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


def test_df_modelresult(klagshamn):
    df = klagshamn.df
    df["ones"] = 1.0

    mr1 = DataFrameModelResult(df, item=0)
    assert isinstance(mr1, ModelResultInterface)
    assert mr1.start_time == datetime(2015, 1, 1, 1, 0, 0)
    assert mr1.end_time == datetime(2020, 9, 28, 0, 0, 0)
    assert mr1.name == "Water Level"

    mr2 = DataFrameModelResult(df[["Water Level"]])
    assert len(mr2.df) == len(mr1.df)

    # Series
    mr3 = DataFrameModelResult(df["Water Level"])
    assert len(mr3.df) == len(mr1.df)

    # item as string
    mr4 = DataFrameModelResult(df, item="Water Level")
    assert len(mr4.df) == len(mr1.df)


def test_repr(hd_oresund_2d):
    mr = ModelResult(hd_oresund_2d)
    txt = repr(mr)
    assert "Oresund2D.dfsu" in txt


def test_dfs_object(hd_oresund_2d):
    mr = ModelResult(hd_oresund_2d)

    assert mr.dfs.is_2d


def test_ModelResultType(sw_dutch_coast):
    mr = ModelResult(sw_dutch_coast)

    assert mr.is_dfsu


def test_ModelResultType0():
    mr = ModelResult("tests/testdata/TS.dfs0")

    assert mr.is_dfs0


# def test_extract_observation(sw_dutch_coast, Hm0_HKNA):
#     mr = ModelResult(sw_dutch_coast)
#     c = mr.extract_observation(Hm0_HKNA)  # infer item by EUM
#     assert c.n_points == 386


def test_extract_observation_no_matching_item(sw_total_windsea, wind_HKNA):
    mr = ModelResult(sw_total_windsea)  # No wind speed here !

    with pytest.raises(Exception):  # More specific error?
        _ = mr.extract_observation(wind_HKNA)


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

    # with pytest.raises(Exception):
    #     c = mr.extract_observation(Hm0_HKNA)  # infer item by EUM is ambigous

    # Specify Swell item explicitely
    c = mr["Sign. Wave Height, S"].extract_observation(Hm0_HKNA)
    assert c.n_points > 0


def test_extract_observation_validation(hd_oresund_2d, klagshamn):
    mr = ModelResult(hd_oresund_2d)
    with pytest.raises(Exception):
        c = mr[0].extract_observation(klagshamn, validate=True)

    c = mr[0].extract_observation(klagshamn, validate=False)
    assert c.n_points > 0


def test_extract_observation_outside(hd_oresund_2d, klagshamn):
    mr = ModelResult(hd_oresund_2d)
    # correct eum, but outside domain
    klagshamn.itemInfo = eum.ItemInfo(eum.EUMType.Surface_Elevation)
    klagshamn.y = -10
    with pytest.raises(ValueError):
        _ = mr[0].extract_observation(klagshamn, validate=True)


from fmskill.model import DfsModelResultItem, DfsModelResult  # , ModelResultFactory


def test_dfs_model_result(hd_oresund_2d):
    mr = DfsModelResult(hd_oresund_2d, "Oresund")
    assert mr.n_items == 7
    assert isinstance(mr, DfsModelResult)

    mr0 = mr[0]
    assert isinstance(mr0, DfsModelResultItem)
    assert mr.item_names[0] == mr0.item_name

    mr1 = mr["Surface elevation"]
    assert mr.item_names[0] == mr1.item_name
    assert mr.filename == mr1.filename
    assert mr.name == mr1.name


def test_factory(hd_oresund_2d):
    mr = ModelResult(hd_oresund_2d, name="myname")
    assert isinstance(mr, DfsModelResult)
    assert mr.name == "myname"
    assert mr.n_items == 7

    mri = ModelResult(hd_oresund_2d, item="Surface elevation")
    assert isinstance(mri, DfsModelResultItem)
    assert mri.item_name == "Surface elevation"
