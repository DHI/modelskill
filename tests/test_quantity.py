import pytest
import modelskill as ms


def test_str():
    wh = ms.Quantity(name="Significant wave height", unit="m")
    assert str(wh) == "Significant wave height [m]"


def test_from_EUMType_string():
    with pytest.warns(match="unit"):
        # mikeio.EUMType.Significant_wave_height
        q = ms.Quantity.from_mikeio_eum_name("Significant_wave_height")

    assert q.unit == "m"

    with pytest.warns(match="unit"):
        # mikeio.EUMType.Discharge
        q = ms.Quantity.from_mikeio_eum_name("Discharge")

    assert q.unit == "m^3/s"


def test_unknown_quantity_raises_error():
    with pytest.raises(ValueError):
        ms.Quantity.from_mikeio_eum_name("foo")


def test_from_cf_attrs():
    q = ms.Quantity.from_cf_attrs({"long_name": "Wind speed", "units": "meter"})
    assert q.name == "Wind speed"
    assert q.unit == "meter"
    assert not q.is_directional


def test_from_cf_attrs_directional():
    q = ms.Quantity.from_cf_attrs({"long_name": "Wind direction", "units": "degree"})
    assert q.name == "Wind direction"
    assert q.unit == "degree"
    assert q.is_directional


def test_from_cf_attrs_incomplete():
    q = ms.Quantity.from_cf_attrs({"long_name": "Wind speed"})
    assert q.name == ""
    assert q.unit == ""


def test_is_compatible_ignores_name():
    wl = ms.Quantity(name="Water Level", unit="m")
    assert wl.is_compatible(ms.Quantity(name="Surface Elevation", unit="m"))


def test_is_compatible_different_units():
    wl = ms.Quantity(name="Water Level", unit="m")
    assert not wl.is_compatible(ms.Quantity(name="Discharge", unit="m^3/s"))


def test_is_compatible_compares_units_as_written():
    wl = ms.Quantity(name="Water Level", unit="meter")
    assert not wl.is_compatible(ms.Quantity(name="Water Level", unit="m"))


@pytest.mark.parametrize(
    "other",
    [
        ms.Quantity.undefined(),
        ms.Quantity(name="Undefined", unit="Undefined"),
        ms.Quantity(name="Undefined", unit="undefined"),
        ms.Quantity(name="Pressure", unit=""),
    ],
)
def test_is_compatible_without_unit(other):
    q = ms.Quantity(name="Pressure", unit="MetresWater")
    assert q.is_compatible(other)
    assert other.is_compatible(q)
