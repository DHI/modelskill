import pytest
import modelskill as ms


def test_str():
    wh = ms.Quantity(name="Significant wave height", unit="m")
    assert str(wh) == "Significant wave height [m]"


def test_from_EUMType_string():
    with pytest.warns(match="unit"):
        # mikeio.EUMType.Significant_wave_height
        q = ms.Quantity.from_mikeio_eum_name("Significant_wave_height")

    assert q.unit == "meter"

    with pytest.warns(match="unit"):
        # mikeio.EUMType.Discharge
        q = ms.Quantity.from_mikeio_eum_name("Discharge")

    # TODO should this be "meter^3 per second"? or m3/s?
    assert q.unit == "meter_pow_3_per_sec"


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
