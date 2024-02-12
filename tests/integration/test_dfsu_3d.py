import pytest
import modelskill as ms


def test_dfsu_3d_sigma_z_can_be_used_with_point_obs():
    """Extracting data from a dfsu3d can be very slow, but it should still work."""
    obs = ms.PointObservation(
        "tests/testdata/oresund_sigma_z_340000_6150000.dfs0",
        item="Temperature",
        x=340000,
        y=6150000,
        z=-10,
        name="Koge bugt",
    )
    mr = ms.DfsuModelResult(
        "tests/testdata/oresund_sigma_z.dfsu", item="Temperature", name="Sound"
    )

    # TODO raise error if spatial_method is not supported
    cmp = ms.match(obs=obs, mod=mr, spatial_method="contained")

    assert cmp.score()["Sound"] == pytest.approx(0.13158414083745135)


def test_dfsu_3d_interpolation_not_supported():
    """Extracting data from a dfsu3d can be very slow, but it should still work."""
    obs = ms.PointObservation(
        "tests/testdata/oresund_sigma_z_340000_6150000.dfs0",
        item="Temperature",
        x=340000,
        y=6150000,
        z=-10,
        name="Koge bugt",
    )
    mr = ms.DfsuModelResult(
        "tests/testdata/oresund_sigma_z.dfsu", item="Temperature", name="Sound"
    )

    with pytest.raises(NotImplementedError):
        ms.match(obs=obs, mod=mr, spatial_method="inverse_distance")
