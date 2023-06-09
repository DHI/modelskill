from modelskill.observation import unit_display_name


def test_units_display_name():

    assert unit_display_name("meter") == "m"
    assert unit_display_name("meter_per_sec") == "m/s"
    assert unit_display_name("second") == "s"
