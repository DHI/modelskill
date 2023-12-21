import numpy as np
import pytest

import mikeio
import modelskill as ms


@pytest.fixture
def hd_oresund_2d():
    return "tests/testdata/Oresund2D_subset.dfsu"


# TODO: replace with shorter dfs0
@pytest.fixture
def klagshamn():
    fn = "tests/testdata/smhi_2095_klagshamn.dfs0"
    return ms.PointObservation(fn, item=0, x=366844, y=6154291, name="Klagshamn")


@pytest.fixture
def drogden():
    # >>> from pyproj import Transformer
    # >>> t = Transformer.from_crs(4326,32633, always_xy=True)
    # >>> t.transform(12.7113,55.5364)
    # (355568.6130331255, 6156863.0187071245)

    fn = "tests/testdata/dmi_30357_Drogden_Fyr.dfs0"
    return ms.PointObservation(fn, item=0, x=355568.0, y=6156863.0)


@pytest.fixture
def sw_dutch_coast():
    return "tests/testdata/SW/DutchCoast_2017_subset.dfsu"


@pytest.fixture
def sw_total_windsea():
    return "tests/testdata/SW/SW_Tot_Wind_Swell.dfsu"


@pytest.fixture
def Hm0_HKNA():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return ms.PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def wind_HKNA():
    fn = "tests/testdata/SW/HKNA_wind.dfs0"
    return ms.PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def Hm0_EPL():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return ms.PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def Hm0_C2():
    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    return ms.TrackObservation(fn, item=3, name="C2")


def test_dfsu_repr(hd_oresund_2d):
    mr = ms.model_result(hd_oresund_2d, name="Oresund2D", item="Surface elevation")
    txt = repr(mr)
    assert "Oresund2D" in txt


def test_dfsu_properties(hd_oresund_2d):
    mr = ms.model_result(hd_oresund_2d, name="Oresund2d", item="Surface elevation")

    assert mr.data.is_2d

    # Note != name of item
    assert mr.quantity.name == "Surface Elevation"
    assert mr.quantity.unit == "m"


def test_dfsu_sw(sw_dutch_coast):
    mr = ms.model_result(sw_dutch_coast, name="SW", item=0)

    assert isinstance(mr, ms.DfsuModelResult)


def test_dfsu_aux_items(hd_oresund_2d):
    mr = ms.DfsuModelResult(hd_oresund_2d, item=0, aux_items=["U velocity"])
    assert mr.sel_items.values == "Surface elevation"
    assert mr.sel_items.aux == ["U velocity"]

    mr = ms.DfsuModelResult(
        hd_oresund_2d, item=0, aux_items=["U velocity", "V velocity"]
    )
    assert mr.sel_items.values == "Surface elevation"
    assert mr.sel_items.aux == ["U velocity", "V velocity"]

    # accept string instead of list
    mr = ms.DfsuModelResult(hd_oresund_2d, item=0, aux_items="U velocity")
    assert mr.sel_items.values == "Surface elevation"
    assert mr.sel_items.aux == ["U velocity"]

    # use index instead of name
    mr = ms.DfsuModelResult(hd_oresund_2d, item=0, aux_items=[2, 3])
    assert mr.sel_items.values == "Surface elevation"
    assert mr.sel_items.aux == ["U velocity", "V velocity"]


def test_dfsu_aux_items_fail(hd_oresund_2d):
    with pytest.raises(ValueError, match="Duplicate items"):
        ms.DfsuModelResult(
            hd_oresund_2d, item=0, aux_items=["U velocity", "Surface elevation"]
        )

    with pytest.raises(ValueError, match="Duplicate items"):
        ms.DfsuModelResult(
            hd_oresund_2d, item=0, aux_items=["U velocity", "Surface elevation"]
        )


def test_dfsu_dataarray(hd_oresund_2d):
    ds = mikeio.read(hd_oresund_2d)
    assert ds.n_items == 4
    da = ds[0]
    assert isinstance(da, mikeio.DataArray)

    mr = ms.model_result(da, name="Oresund")
    assert mr.name == "Oresund"
    assert isinstance(mr.data, mikeio.DataArray)

    mr.name = "Oresund2"
    assert mr.name == "Oresund2"


def test_dfsu_factory(hd_oresund_2d):
    mr1 = ms.model_result(hd_oresund_2d, name="myname", item=-1)
    assert isinstance(mr1, ms.DfsuModelResult)
    assert mr1.name == "myname"

    mr2 = ms.model_result(hd_oresund_2d, name="Oresund2d", item="Surface elevation")
    assert isinstance(mr2, ms.DfsuModelResult)
    assert mr2.name == "Oresund2d"


# def test_extract_observation(sw_dutch_coast, Hm0_HKNA):
#     mr = ModelResult(sw_dutch_coast)
#     c = mr.extract_observation(Hm0_HKNA)  # infer item by EUM
#     assert c.n_points == 386


# def test_extract_observation_no_matching_item(sw_total_windsea, wind_HKNA):
#     mr = ModelResult(sw_total_windsea)  # No wind speed here !

#     with pytest.raises(Exception):  # More specific error?
#         _ = mr.extract_observation(wind_HKNA)


def test_extract_observation_total_windsea_swell_not_possible(
    sw_total_windsea, Hm0_HKNA
):
    mr = ms.model_result(sw_total_windsea, name="SW", item="Sign. Wave Height, S")
    """
    Items:
        0:  Sign. Wave Height <Significant wave height> (meter)
        1:  Sign. Wave Height, W <Significant wave height> (meter)
        2:  Sign. Wave Height, S <Significant wave height> (meter)
    """

    # with pytest.raises(Exception):
    #     c = mr.extract_observation(Hm0_HKNA)  # infer item by EUM is ambigous

    # Specify Swell item explicitely
    cc = ms.match(Hm0_HKNA, mr)
    assert cc.n_points > 0


def test_extract_observation_validation(hd_oresund_2d, klagshamn):
    mr = ms.model_result(hd_oresund_2d, item=0)
    with pytest.raises(Exception):
        with pytest.warns(FutureWarning, match="modelskill.match"):
            _ = ms.Connector(klagshamn, mr, validate=True).extract()

    # No error if validate==False
    with pytest.warns(FutureWarning, match="modelskill.match"):
        con = ms.Connector(klagshamn, mr, validate=False)

    c = con.extract()
    assert c.n_points > 0


def test_extract_observation_outside(hd_oresund_2d, klagshamn):
    mr = ms.model_result(hd_oresund_2d, item=0)
    # correct eum, but outside domain
    klagshamn.y = -10
    with pytest.raises(ValueError):
        with pytest.warns(FutureWarning, match="modelskill.match"):
            _ = ms.Connector(klagshamn, mr, validate=True).extract()


def test_dfsu_extract_point(sw_dutch_coast, Hm0_EPL):
    mr1 = ms.model_result(sw_dutch_coast, item=0, name="SW1")
    mr_extr_1 = mr1.extract(Hm0_EPL.copy())
    # df1 = mr1._extract_point(Hm0_EPL)
    assert list(mr_extr_1.data.data_vars) == ["SW1"]
    assert mr_extr_1.n_points == 23

    da = mikeio.read(sw_dutch_coast)[0]
    mr2 = ms.model_result(da, name="SW1")
    mr_extr_2 = mr2.extract(Hm0_EPL.copy())

    assert list(mr_extr_1.data.data_vars) == list(mr_extr_2.data.data_vars)
    assert np.all(mr_extr_1.data == mr_extr_2.data)

    c1 = mr1.extract(Hm0_EPL.copy())
    c2 = mr2.extract(Hm0_EPL.copy())
    assert isinstance(c1, ms.PointModelResult)
    assert isinstance(c2, ms.PointModelResult)
    assert np.all(c1.data == c2.data)
    # c1.observation.itemInfo == Hm0_EPL.itemInfo
    # assert len(c1.observation.data.index.difference(Hm0_EPL.data.index)) == 0


def test_dfsu_extract_point_aux(sw_dutch_coast, Hm0_EPL):
    mr1 = ms.model_result(
        sw_dutch_coast, item=0, aux_items=["Peak Wave Direction"], name="SW1"
    )
    mr_extr_1 = mr1.extract(Hm0_EPL.copy())
    assert list(mr_extr_1.data.data_vars) == ["SW1", "Peak Wave Direction"]
    assert mr_extr_1.n_points == 23


def test_dfsu_extract_track(sw_dutch_coast, Hm0_C2):
    mr1 = ms.model_result(sw_dutch_coast, item=0, name="SW1")
    mr_track1 = mr1.extract(Hm0_C2)
    ds1 = mr_track1.data
    assert "SW1" in ds1.data_vars
    assert "x" in ds1.coords
    assert "y" in ds1.coords
    assert mr_track1.n_points == 70

    da = mikeio.read(sw_dutch_coast)[0]
    mr2 = ms.model_result(da, name="SW1")
    mr_track2 = mr2.extract(Hm0_C2.copy())
    ds2 = mr_track2.data

    assert list(ds1.data_vars) == list(ds2.data_vars)
    assert np.all(ds1 == ds2)

    c1 = mr1.extract(Hm0_C2.copy())
    c2 = mr2.extract(Hm0_C2.copy())
    assert isinstance(c1, ms.TrackModelResult)
    assert isinstance(c2, ms.TrackModelResult)
    assert np.all(c1.data == c2.data)
    # c1.observation.itemInfo == Hm0_C2.itemInfo
    # assert len(c1.observation.data.index.difference(Hm0_C2.data.index)) == 0


def test_dfsu_extract_track_aux(sw_dutch_coast, Hm0_C2):
    mr1 = ms.model_result(
        sw_dutch_coast, item=0, aux_items=["Peak Wave Direction"], name="SW1"
    )
    mr_track1 = mr1.extract(Hm0_C2)
    assert list(mr_track1.data.data_vars) == ["SW1", "Peak Wave Direction"]
    assert "x" in mr_track1.data.coords
    assert "y" in mr_track1.data.coords
    assert mr_track1.n_points == 70
