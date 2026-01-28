import modelskill as ms
import pytest
import xarray as xr


def test_extract_point_from_3d():
    mr = ms.GridModelResult(
        "tests/testdata/cmems_mod_med_phy-sal_anfc_4.2km-3D_PT1H-m_1705916517624.nc",
        item="so",
        name="MedSea",
    )

    point_ds = xr.open_dataset("tests/testdata/aegean_sea_salinity_ts.nc")

    # TODO use x,y,z without explicitly setting them (NetCDF has them as coordinates)
    obs = ms.PointObservation(
        point_ds,
        x=float(point_ds.longitude),
        y=float(point_ds.latitude),
        z=float(point_ds.depth),
        item="so",
    )

    cmp = ms.match(obs=obs, mod=mr, spatial_method="nearest")
    assert cmp.quantity.name == "Salinity"
    assert cmp.quantity.unit == "0.001"

    sc = cmp.score()

    # "Observed" data is extracted from the 3D model result, so the score should be 0.0
    assert sc["MedSea"] == pytest.approx(0.0)
