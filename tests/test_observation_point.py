import pandas as pd
import pytest
import mikeio

from modelskill.observation import PointObservation
from modelskill import Quantity


@pytest.fixture
def klagshamn_filename():
    return "tests/testdata/smhi_2095_klagshamn.dfs0"


@pytest.fixture
def klagshamn_df(klagshamn_filename):
    return mikeio.read(klagshamn_filename).to_dataframe()


@pytest.fixture
def klagshamn_da(klagshamn_filename):
    da = mikeio.read(klagshamn_filename)["Water Level"]
    assert isinstance(da, mikeio.DataArray)
    return da


@pytest.fixture
def klagshamn_ds(klagshamn_filename):
    return mikeio.read(klagshamn_filename)


def test_from_dfs0(klagshamn_filename):
    o1 = PointObservation(
        klagshamn_filename, item=0, x=366844, y=6154291, name="Klagshamn"
    )
    assert o1.n_points == 50328

    o2 = PointObservation(klagshamn_filename, item="Water Level", x=366844, y=6154291)
    assert o1.n_points == o2.n_points

    o3 = PointObservation(
        klagshamn_filename,
        item="Water Level",
        x=366844,
        y=6154291,
        quantity=Quantity(name="Water level", unit="meter"),
    )

    assert o3.quantity.unit == "meter"

    o6 = PointObservation(
        klagshamn_filename,
        item="Water Level",
        x=366844,
        y=6154291,
        quantity=Quantity(name="Water level", unit="feet"),
    )

    assert o6.quantity.unit == "feet"


def test_from_df_quantity_from_string(klagshamn_filename):
    o1 = PointObservation(
        klagshamn_filename,
        item=0,
        x=366844,
        y=6154291,
        name="Klagshamn1",
        quantity="Water_Level",  # TODO is this intuitive ?
    )

    assert o1.quantity.unit == "meter"


def test_from_df_quantity_from_string_without_underscore(klagshamn_filename):
    o1 = PointObservation(
        klagshamn_filename,
        item=0,
        x=366844,
        y=6154291,
        name="Klagshamn1",
        quantity="Water Level",  # TODO is this intuitive ?
    )

    assert o1.quantity.unit == "meter"


def test_from_mikeio_dataarray(klagshamn_da):
    o = PointObservation(klagshamn_da, x=366844, y=6154291, name="Klagshamn")
    assert o.quantity.name == "Water Level"
    assert o.quantity.unit == "meter"


def test_from_mikeio_dataarray_with_quantity(klagshamn_da):
    o = PointObservation(
        klagshamn_da,
        x=366844,
        y=6154291,
        name="Klagshamn",
        quantity=Quantity(name="Niveau", unit="fathoms"),
    )
    assert o.quantity.name == "Niveau"
    assert o.quantity.unit == "fathoms"


def test_from_mikeio_dataset(klagshamn_ds):
    o = PointObservation(
        klagshamn_ds, item="Water Level", x=366844, y=6154291, name="Klagshamn"
    )
    assert o.quantity.name == "Water Level"
    assert o.quantity.unit == "meter"


def test_from_df(klagshamn_filename, klagshamn_df):
    o1 = PointObservation(
        klagshamn_filename, item=0, x=366844, y=6154291, name="Klagshamn1"
    )

    df = klagshamn_df
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


def test_hist(klagshamn_filename):
    o1 = PointObservation(
        klagshamn_filename, item=0, x=366844, y=6154291, name="Klagshamn1"
    )
    o1.plot.hist()
    o1.plot.hist(density=False)
    o1.plot.hist(bins=20, title="new_title", color="red")


def test_force_keyword_args(klagshamn_filename):

    with pytest.raises(TypeError):
        PointObservation(klagshamn_filename, 0, 366844, 6154291, "Klagshamn")
