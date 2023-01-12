import pytest

import mikeio
from fmskill.model import ModelResult
from fmskill.model import DfsModelResultItem, DfsModelResult,DataArrayModelResultItem
from fmskill.observation import PointObservation,TrackObservation
from fmskill.comparison import PointComparer, TrackComparer

import numpy as np



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


@pytest.fixture
def sw_dutch_coast():
    return "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"


@pytest.fixture
def sw_total_windsea():
    return "tests/testdata/SW/SW_Tot_Wind_Swell.dfsu"


@pytest.fixture
def sw_Hm0_df():
    fn = "tests/testdata/SW/ts_storm_4.dfs0"
    return mikeio.read(fn, items=0).to_dataframe()


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
        with pytest.warns(UserWarning, match="Item type should match"):
            c = mr[0].extract_observation(klagshamn, validate=True)

    c = mr[0].extract_observation(klagshamn, validate=False)
    assert c.n_points > 0


def test_extract_observation_outside(hd_oresund_2d, klagshamn):
    mr = ModelResult(hd_oresund_2d)
    # correct eum, but outside domain
    klagshamn.itemInfo = mikeio.ItemInfo(mikeio.EUMType.Surface_Elevation)
    klagshamn.y = -10
    with pytest.raises(ValueError):
        _ = mr[0].extract_observation(klagshamn, validate=True)


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


def test_dataarray_model_result(hd_oresund_2d):
    ds = mikeio.read(hd_oresund_2d)
    assert ds.n_items == 7
    da = ds[0]
    assert isinstance(da, mikeio.DataArray)

    mr = ModelResult(da, name="Oresund")
    assert isinstance(mr, DataArrayModelResultItem)
    assert mr.item_name == da.item.name
    assert mr.name == "Oresund"
    assert isinstance(mr._da, mikeio.DataArray)

    mr.name = "Oresund2"
    assert mr.name == "Oresund2"


def test_dataarray_extract_point(sw_dutch_coast, Hm0_EPL):
    mr1 = ModelResult(sw_dutch_coast, item=0, name="SW1")
    assert mr1.itemInfo.type == mikeio.EUMType.Significant_wave_height
    df1 = mr1._extract_point(Hm0_EPL)
    assert df1.columns == ["SW1"]
    assert len(df1) == 23

    da = mikeio.read(sw_dutch_coast)[0]
    mr2 = ModelResult(da, name="SW1")
    assert mr2.itemInfo.type == mikeio.EUMType.Significant_wave_height
    df2 = mr2._extract_point(Hm0_EPL.copy())

    assert df1.columns == df2.columns
    assert np.all(df1 == df2)

    c1 = mr1.extract_observation(Hm0_EPL.copy())
    c2 = mr2.extract_observation(Hm0_EPL.copy())
    assert isinstance(c1, PointComparer)
    assert isinstance(c2, PointComparer)
    assert np.all(c1.df == c2.df)
    c1.observation.itemInfo == Hm0_EPL.itemInfo
    assert len(c1.observation.df.index.difference(Hm0_EPL.df.index)) == 0


def test_dataarray_extract_track(sw_dutch_coast, Hm0_C2):
    mr1 = ModelResult(sw_dutch_coast, item=0, name="SW1")
    df1 = mr1._extract_track(Hm0_C2)
    assert list(df1.columns) == ["Longitude", "Latitude", "SW1"]
    assert len(df1) == 113

    da = mikeio.read(sw_dutch_coast)[0]
    mr2 = ModelResult(da, name="SW1")
    df2 = mr2._extract_track(Hm0_C2.copy())

    assert list(df1.columns) == list(df2.columns)
    assert np.all(df1 == df2)

    c1 = mr1.extract_observation(Hm0_C2.copy())
    c2 = mr2.extract_observation(Hm0_C2.copy())
    assert isinstance(c1, TrackComparer)
    assert isinstance(c2, TrackComparer)
    assert np.all(c1.df == c2.df)
    c1.observation.itemInfo == Hm0_C2.itemInfo
    assert len(c1.observation.df.index.difference(Hm0_C2.df.index)) == 0


def test_factory(hd_oresund_2d):
    mr = ModelResult(hd_oresund_2d, name="myname")
    assert isinstance(mr, DfsModelResult)
    assert mr.name == "myname"
    assert mr.n_items == 7

    mri = ModelResult(hd_oresund_2d, item="Surface elevation")
    assert isinstance(mri, DfsModelResultItem)
    assert mri.item_name == "Surface elevation"
