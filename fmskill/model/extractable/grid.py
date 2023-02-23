from typing import Union

import xarray as xr

from fmskill.model import protocols
from fmskill.model._base import ModelResultBase
from fmskill.observation import PointObservation, TrackObservation


class GridModelResult(ModelResultBase):
    def extract(self, observation: Union[PointObservation, TrackObservation]):
        pass


if __name__ == "__main__":
    grid_data = xr.open_dataset("tests/testdata/SW/CMEMS_DutchCoast_2017-10-28.nc")
    test = GridModelResult(grid_data, "test", "test", "test")

    assert isinstance(test, protocols.ModelResult)
    assert isinstance(test, protocols.Extractable)
