from datetime import datetime
import xarray as xr


from fmskill.model.xarray2 import XArrayModelResult
from fmskill.observation import PointObservation


def test_XArrayModelResult_from_nc():
    mr = XArrayModelResult.create_from_file("tests/testdata/SW/ERA5_DutchCoast.nc")

    assert isinstance(mr.ds, xr.Dataset)
    # assert "- Item: 4: swh" in repr(mr)
    assert len(mr) == 5
    assert len(mr.ds) == 5
    assert mr.name == "ERA5_DutchCoast"
    assert mr.item_names == ["mwd", "mwp", "mp2", "pp1d", "swh"]
    assert mr.start_time == datetime(2017, 10, 27, 0, 0, 0)
    assert mr.end_time == datetime(2017, 10, 29, 18, 0, 0)


def test_XArrayModelResult_extract_point():
    obs = PointObservation(
        "tests/testdata/SW/eur_Hm0.dfs0", item=0, x=3.2760, y=51.9990, name="EPL"
    )
    mr = XArrayModelResult.create_from_file("tests/testdata/SW/ERA5_DutchCoast.nc")
    cmp = mr.extract_observation(obs, item="swh")
    df = cmp.df
    assert len(df.columns) == 2
