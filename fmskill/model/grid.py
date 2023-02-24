from typing import Union

import xarray as xr
import mikeio

from fmskill.model import protocols, extraction
from fmskill.model._base import ModelResultBase
from fmskill.observation import PointObservation, TrackObservation


class GridModelResult(ModelResultBase):
    def extract(
        self, observation: Union[PointObservation, TrackObservation]
    ) -> protocols.Comparable:
        type_extraction_mapping = {
            (xr.Dataset, PointObservation): extraction.point_obs_from_xr_mr,
            (xr.Dataset, TrackObservation): extraction.track_obs_from_xr_mr,
            (
                mikeio.Dataset,
                PointObservation,
            ): lambda mr, o: extraction.point_obs_from_xr_mr(mr.to_xarray(), o),
            (
                mikeio.Dataset,
                TrackObservation,
            ): lambda mr, o: extraction.track_obs_from_xr_mr(mr.to_xarray(), o),
            (mikeio.Dfs2, PointObservation): None,  # Possible future work
            (mikeio.Dfs2, TrackObservation): None,  # Possible future work
        }

        extraction_func = type_extraction_mapping.get(
            (type(self.data), type(observation))
        )
        if extraction_func is None:
            raise NotImplementedError(
                f"Extraction from {type(self.data)} to {type(observation)} is not implemented."
            )
        extraction_result = extraction_func(self, observation)

        return extraction_result


if __name__ == "__main__":
    grid_data = xr.open_dataset("tests/testdata/SW/ERA5_DutchCoast.nc")
    point_obs = PointObservation(
        "tests/testdata/SW/eur_Hm0.dfs0", item=0, x=3.2760, y=51.9990, name="EPL"
    )
    track_obs = TrackObservation(
        "tests/testdata/SW/Alti_c2_Dutch.dfs0", item=3, name="c2"
    )
    test = GridModelResult(grid_data, item="swh", name="test")

    assert isinstance(test, protocols.ModelResult)
    assert isinstance(test, protocols.Extractable)

    test.extract(point_obs)
    test.extract(track_obs)
