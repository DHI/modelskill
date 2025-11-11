import modelskill as ms


def test_network_observation():
    obs = ms.NetworkLocationObservation(
        "tests/testdata/network/vidaa_mag_4905.dfs0",
        item=0,
        reach="VIDAA-MAG",
        chainage="4905",
    )
    assert obs.quantity.name == "Discharge"
    assert obs.quantity.unit == "m^3/s"
    assert obs.reach == "VIDAA-MAG"
    assert obs.chainage == "4905"
