import pandas as pd
import pytest
import sys
import mikeio

from fmskill.observation import PointObservation
from fmskill import Quantity


@pytest.fixture
def klagshamn():
    return "tests/testdata/smhi_2095_klagshamn.dfs0"


def test_from_dfs0(klagshamn):
    o1 = PointObservation(klagshamn, item=0, x=366844, y=6154291, name="Klagshamn")
    assert o1.n_points == 50328

    o2 = PointObservation(klagshamn, item="Water Level", x=366844, y=6154291)
    assert o1.n_points == o2.n_points

    o3 = PointObservation(
        klagshamn,
        item="Water Level",
        x=366844,
        y=6154291,
        quantity=Quantity(name="Water level", unit="meter"),
    )

    assert o3.quantity.unit == "meter"

    o6 = PointObservation(
        klagshamn,
        item="Water Level",
        x=366844,
        y=6154291,
        quantity=Quantity(name="Water level", unit="feet"),
    )

    assert o6.quantity.unit == "feet"


def test_from_df_quantity_from_string(klagshamn):
    o1 = PointObservation(
        klagshamn,
        item=0,
        x=366844,
        y=6154291,
        name="Klagshamn1",
        quantity="Water_Level",  # TODO is this intuitive ?
    )

    assert o1.quantity.unit == "meter"


def test_from_df_quantity_from_string_without_underscore(klagshamn):
    o1 = PointObservation(
        klagshamn,
        item=0,
        x=366844,
        y=6154291,
        name="Klagshamn1",
        quantity="Water Level",  # TODO is this intuitive ?
    )

    assert o1.quantity.unit == "meter"


def test_from_df(klagshamn):
    o1 = PointObservation(klagshamn, item=0, x=366844, y=6154291, name="Klagshamn1")

    df = o1.data
    assert isinstance(df, pd.DataFrame)
    o2 = PointObservation(df, item=0, x=366844, y=6154291, name="Klagshamn2")
    assert o1.n_points == o2.n_points

    # item as str
    o2 = PointObservation(df, item="Water Level", x=366844, y=6154291)
    assert o1.n_points == o2.n_points

    s = o1.data["Water Level"]
    assert isinstance(s, pd.Series)
    o3 = PointObservation(s, x=366844, y=6154291, name="Klagshamn3")
    assert o1.n_points == o3.n_points


def test_hist(klagshamn):
    o1 = PointObservation(klagshamn, item=0, x=366844, y=6154291, name="Klagshamn1")
    o1.plot.hist()
    o1.plot.hist(density=False)
    o1.plot.hist(bins=20, title="new_title", color="red")


def test_force_keyword_args(klagshamn):

    with pytest.raises(TypeError):
        PointObservation(klagshamn, 0, 366844, 6154291, "Klagshamn")
