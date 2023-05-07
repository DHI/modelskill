from fmskill import Quantity


def test_str():
    wh = Quantity(name="Significant wave height", unit="m")
    assert str(wh) == "Significant wave height [m]"
