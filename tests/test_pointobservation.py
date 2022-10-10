import pandas as pd
import pytest
import sys
import mikeio

from fmskill.observation import PointObservation


@pytest.fixture
def klagshamn():
    return "tests/testdata/smhi_2095_klagshamn.dfs0"


def test_from_dfs0(klagshamn):
    o1 = PointObservation(klagshamn, item=0, x=366844, y=6154291, name="Klagshamn")
    assert o1.n_points == 50328

    o2 = PointObservation(klagshamn, item="Water Level", x=366844, y=6154291)
    assert o1.n_points == o2.n_points

    o3 = PointObservation(klagshamn, item="Water Level", x=366844, y=6154291, units='meter')
    assert o3.override_units == o2.itemInfo.unit.name

    o4 = PointObservation(klagshamn, item="Water Level", x=366844, y=6154291)
    assert o4.override_units == None

    o5 = PointObservation(klagshamn, item="Water Level", x=366844, y=6154291)
    assert o5._unit_text() == 'Water Level [m]'

    o6 = PointObservation(klagshamn, item="Water Level", x=366844, y=6154291, units='inches')
    assert o6._unit_text() == "Water Level [inches]"


def test_from_df(klagshamn):
    o1 = PointObservation(klagshamn, item=0, x=366844, y=6154291, name="Klagshamn1")

    df = o1.df
    assert isinstance(df, pd.DataFrame)
    o2 = PointObservation(df, item=0, x=366844, y=6154291, name="Klagshamn2")
    assert o1.n_points == o2.n_points

    # item as str
    o2 = PointObservation(df, item="Water Level", x=366844, y=6154291)
    assert o1.n_points == o2.n_points

    s = o1.df["Water Level"]
    assert isinstance(s, pd.Series)
    o3 = PointObservation(s, x=366844, y=6154291, name="Klagshamn3")
    assert o1.n_points == o3.n_points

    o4 = PointObservation(df, item="Water Level", x=366844, y=6154291, units='metre')
    assert o4.override_units == 'metre'

    o5 = PointObservation(df, item="Water Level", x=366844, y=6154291, units='inches')
    o5.itemInfo = mikeio.ItemInfo(mikeio.EUMType.Water_Level) 
    assert o5._unit_text() == "Water Level [inches]"


@pytest.mark.skipif("shapely" not in sys.modules, reason="requires the shapely")
def test_coordinates(klagshamn):
    from shapely.geometry import Point

    x0, y0 = 366844, 6154291
    o1 = PointObservation(klagshamn, item=0, x=x0, y=y0, name="Klagshamn")
    assert isinstance(o1.geometry, Point)
    assert o1.geometry.x == x0
    assert o1.geometry.y == y0


def test_hist(klagshamn):
    o1 = PointObservation(klagshamn, item=0, x=366844, y=6154291, name="Klagshamn1")
    o1.hist()
    o1.hist(density=False)
    o1.hist(bins=20, title="new_title", color="red")


def test_force_keyword_args(klagshamn):

    with pytest.raises(TypeError):
        PointObservation(klagshamn, 0, 366844, 6154291, "Klagshamn")
