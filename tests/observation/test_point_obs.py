import numpy as np
import pandas as pd
import pytest
import mikeio

import modelskill as ms


@pytest.fixture
def klagshamn_filename():
    return "tests/testdata/smhi_2095_klagshamn_200.dfs0"


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


@pytest.fixture
def df_aux():
    df = pd.DataFrame(
        {
            "WL": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "aux1": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            "aux2": [1.2, 2.2, 3.2, 4.2, 5.2, 6.2],
            "time": pd.date_range("2019-01-01", periods=6, freq="D"),
        }
    ).set_index("time")
    return df


def test_from_dfs0(klagshamn_filename):
    o1 = ms.PointObservation(
        klagshamn_filename, item=0, x=366844, y=6154291, name="Klagshamn"
    )
    assert o1.n_points == 198  # 200 including 2 NaN

    o2 = ms.PointObservation(
        klagshamn_filename, item="Water Level", x=366844, y=6154291
    )
    assert o1.n_points == o2.n_points

    o3 = ms.PointObservation(
        klagshamn_filename,
        item="Water Level",
        x=366844,
        y=6154291,
        quantity=ms.Quantity(name="Water level", unit="meter"),
    )

    assert o3.quantity.unit == "meter"

    o6 = ms.PointObservation(
        klagshamn_filename,
        item="Water Level",
        x=366844,
        y=6154291,
        quantity=ms.Quantity(name="Water level", unit="feet"),
    )

    assert o6.quantity.unit == "feet"


def test_from_mikeio_dataarray(klagshamn_da):
    o = ms.PointObservation(klagshamn_da, x=366844, y=6154291, name="Klagshamn")
    assert o.quantity.name == "Water Level"
    assert o.quantity.unit == "m"


def test_from_mikeio_dataarray_with_quantity(klagshamn_da):
    o = ms.PointObservation(
        klagshamn_da,
        x=366844,
        y=6154291,
        name="Klagshamn",
        quantity=ms.Quantity(name="Niveau", unit="fathoms"),
    )
    assert o.quantity.name == "Niveau"
    assert o.quantity.unit == "fathoms"


def test_from_mikeio_dataset(klagshamn_ds):
    o = ms.PointObservation(
        klagshamn_ds, item="Water Level", x=366844, y=6154291, name="Klagshamn"
    )
    assert o.quantity.name == "Water Level"
    assert o.quantity.unit == "m"


def test_from_df(klagshamn_filename, klagshamn_df):
    o1 = ms.PointObservation(
        klagshamn_filename, item=0, x=366844, y=6154291, name="Klagshamn1"
    )

    df = klagshamn_df
    assert isinstance(df, pd.DataFrame)
    o2 = ms.PointObservation(df, item=0, x=366844, y=6154291, name="Klagshamn2")
    assert o1.n_points == o2.n_points

    # item as str
    o2 = ms.PointObservation(df, item="Water Level", x=366844, y=6154291)
    assert o1.n_points == o2.n_points

    ser = o1.data["Klagshamn1"]
    # assert isinstance(s, pd.Series)
    o3 = ms.PointObservation(ser, x=366844, y=6154291, name="Klagshamn3")
    assert o1.n_points == o3.n_points


def test_observation_factory(klagshamn_da):
    o = ms.observation(klagshamn_da, x=366844, y=6154291, name="Klagshamn")
    assert isinstance(o, ms.PointObservation)

    with pytest.warns(UserWarning, match="Could not guess geometry"):
        o = ms.observation(klagshamn_da, name="Klagshamn")
        assert isinstance(o, ms.PointObservation)


def test_hist(klagshamn_filename):
    o1 = ms.PointObservation(
        klagshamn_filename, item=0, x=366844, y=6154291, name="Klagshamn1"
    )
    o1.plot.hist()
    o1.plot.hist(density=False)
    o1.plot.hist(bins=20, title="new_title", color="red")


def test_force_keyword_args(klagshamn_filename):
    with pytest.raises(TypeError):
        ms.PointObservation(klagshamn_filename, 0, 366844, 6154291, "Klagshamn")


def test_point_data_can_be_persisted_as_netcdf(klagshamn_filename, tmp_path):
    p = ms.PointObservation(klagshamn_filename)

    p.data.to_netcdf(tmp_path / "test.nc")


def test_attrs(klagshamn_filename):
    o1 = ms.PointObservation(
        klagshamn_filename, item=0, attrs={"a1": "v1"}, name="Klagshamn"
    )
    assert o1.attrs["a1"] == "v1"

    o1.attrs["a2"] = "v2"
    assert o1.attrs["a2"] == "v2"

    o2 = ms.PointObservation(
        klagshamn_filename, item=0, attrs={"version": 42}, name="Klagshamn"
    )
    assert o2.attrs["version"] == 42

    o2.attrs["version"] = 43
    assert o2.attrs["version"] == 43

    # remove all attributes and add a new one
    o2.attrs = {"version": 44}
    assert o2.attrs["version"] == 44


def test_attrs_non_serializable(klagshamn_filename):
    with pytest.raises(ValueError, match="type"):
        ms.PointObservation(
            klagshamn_filename,
            item=0,
            attrs={"related": {"foo": "bar"}},
            name="Klagshamn",
        )


def test_attrs_not_allowed(klagshamn_filename):
    with pytest.raises(ValueError, match="attrs key gtype not allowed"):
        ms.PointObservation(klagshamn_filename, item=0, attrs={"gtype": "v1"})


def test_point_aux_items(df_aux):
    o = ms.PointObservation(df_aux, item="WL", aux_items=["aux1"])
    assert "aux1" in o.data
    assert o.data["aux1"].values[0] == 1.1

    o = ms.PointObservation(df_aux, item="WL", aux_items="aux1")
    assert "aux1" in o.data
    assert o.data["aux1"].values[0] == 1.1


def test_point_aux_items_fail(df_aux):
    with pytest.raises(KeyError):
        ms.PointObservation(df_aux, item="WL", aux_items=["aux1", "aux3"])

    with pytest.raises(ValueError):
        ms.PointObservation(df_aux, item="WL", aux_items="WL")


def test_point_aux_items_multiple(df_aux):
    o = ms.PointObservation(df_aux, item="WL", aux_items=["aux1", "aux2"])
    assert "aux1" in o.data
    assert "aux2" in o.data
    assert o.data["aux1"].values[0] == 1.1
    assert o.data["aux2"].values[0] == 1.2


def test_mikeio_iteminfo_pretty_units():
    da = mikeio.DataArray(
        data=np.array([1, 2, 3]),
        item=mikeio.ItemInfo("Q", mikeio.EUMType.Discharge),
        time=pd.date_range("2019-01-01", periods=3, freq="D"),
    )
    assert da.unit.short_name == "m^3/s"

    obs = ms.PointObservation(da, x=0, y=0)
    assert obs.quantity.unit == "m^3/s"


def test_point_obs_repr(df_aux):
    # Some basic test to see that repr does not fail
    o = ms.PointObservation(df_aux, item="WL", aux_items=["aux1"])
    assert "aux1" in repr(o)

    # TODO ignore this for now
    # o.z = -1
    # assert "-1" in repr(o)


def test_point_observation_without_coords_are_nan():
    # No coords in file, no coords supplied 😳
    obs = ms.PointObservation(
        "tests/testdata/smhi_2095_klagshamn.dfs0", item=0, name="Klagshamn"
    )  #  x=366844, y=6154291,
    assert np.isnan(obs.x)
    assert np.isnan(obs.y)

    # NaN is not the same as None
    assert obs.z is None


def test_point_observation_from_nc_file():
    obs = ms.PointObservation(
        "tests/testdata/smhi_2095_klagshamn.nc", item="Water Level"
    )
    assert obs.x == pytest.approx(366844)
    assert obs.y == pytest.approx(6154291)

    # TODO is using the filename as name a good idea?
    assert obs.name == "smhi_2095_klagshamn"

    named_obs = ms.PointObservation(
        "tests/testdata/smhi_2095_klagshamn.nc",
        item="Water Level",
        name="Klagshamn",
    )

    assert named_obs.name == "Klagshamn"


def test_point_observation_set_coords():
    """Setting x, y explicitly should override the values in the file"""
    obs = ms.PointObservation(
        "tests/testdata/smhi_2095_klagshamn.nc", item="Water Level", x=0, y=0
    )

    assert obs.x == 0
    assert obs.y == 0
    assert obs.z is None

    obs3d = ms.PointObservation(
        "tests/testdata/smhi_2095_klagshamn.nc", item="Water Level", x=0, y=0, z=-5
    )

    assert obs3d.z == -5


def test_point_observation_from_netcdf_with_x_y_name():
    obs = ms.PointObservation("tests/testdata/SW/HKNA_Hm0.nc")

    assert obs.x == pytest.approx(4.242)
    assert obs.y == pytest.approx(52.6887)
    assert obs.name == "HKNA"
    assert obs.quantity.unit == "m"
    assert obs.quantity.name == "Hm0"

    obs2 = ms.PointObservation("tests/testdata/SW/HKNA_Hm0.nc", name="A better name")
    assert obs2.name == "A better name"
