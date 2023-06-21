import pytest
from modelskill import Quantity


def test_str():
    wh = Quantity(name="Significant wave height", unit="m")
    assert str(wh) == "Significant wave height [m]"


def test_from_EUMType_string():

    with pytest.warns(match="unit"):
        # mikeio.EUMType.Significant_wave_height
        q = Quantity.from_mikeio_eum_name("Significant_wave_height")

    assert q.unit == "meter"

    with pytest.warns(match="unit"):
        # mikeio.EUMType.Discharge
        q = Quantity.from_mikeio_eum_name("Discharge")

    # TODO should this be "meter^3 per second"? or m3/s?
    assert q.unit == "meter_pow_3_per_sec"


def test_unknown_quantity_raises_error():
    with pytest.raises(ValueError):
        Quantity.from_mikeio_eum_name("foo")
