from typing import Union

import xarray as xr

from fmskill.model import protocols, extraction
from fmskill.model._base import ModelResultBase
from fmskill.observation import PointObservation, TrackObservation


class GridModelResult(ModelResultBase):
    def extract(self, observation: Union[PointObservation, TrackObservation]):

        type_extraction_mapping = {
            (xr.Dataset, PointObservation): extraction.point_obs_from_xr_mr,
            (xr.Dataset, TrackObservation): extraction.track_obs_from_xr_mr,
        }

        extraction_func = type_extraction_mapping.get(
            (type(self.data), type(observation))
        )
        if extraction_func is None:
            raise NotImplementedError(
                f"Extraction from {type(self.data)} to {type(observation)} is not implemented"
            )
        extraction_result = extraction_func(self, observation)

        return extraction_result


if __name__ == "__main__":
    grid_data = xr.open_dataset("tests/testdata/SW/ERA5_DutchCoast.nc")
    point_obs = PointObservation(
        "tests/testdata/SW/eur_Hm0.dfs0", item=0, x=3.2760, y=51.9990, name="EPL"
    )

    test = GridModelResult(grid_data, item="swh", name="test")

    assert isinstance(test, protocols.ModelResult)
    assert isinstance(test, protocols.Extractable)

    test.extract(point_obs)
