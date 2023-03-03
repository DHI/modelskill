from typing import Union
import warnings

import mikeio
from fmskill.comparison import PointComparer, SingleObsComparer, TrackComparer

from fmskill.model import protocols, extraction
from fmskill.model._base import ModelResultBase
from fmskill.observation import PointObservation, TrackObservation


class DfsuModelResult(ModelResultBase):
    def extract(
        self, observation: Union[PointObservation, TrackObservation]
    ) -> protocols.Comparable:

        type_extraction_mapping = {
            (mikeio.dfsu.Dfsu2DH, PointObservation): extraction.point_obs_from_dfsu_mr,
            (mikeio.dfsu.Dfsu2DH, TrackObservation): extraction.track_obs_from_dfsu_mr,
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

    def extract_observation(
        self, observation: Union[PointObservation, TrackObservation], validate=True
    ) -> SingleObsComparer:
        super().extract_observation(observation, validate)

        point_or_track_mr = self.extract(observation)
        if isinstance(observation, PointObservation):
            comparer = PointComparer(observation, point_or_track_mr.data)
        elif isinstance(observation, TrackObservation):
            comparer = TrackComparer(observation, point_or_track_mr.data)
        else:
            raise ValueError("Only point and track observation are supported!")

        if len(comparer.data) == 0:
            warnings.warn(f"No overlapping data in found for obs '{observation.name}'!")
            comparer = None

        return comparer


if __name__ == "__main__":
    dfsu = mikeio.open("tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu")
    test = DfsuModelResult(
        dfsu,
        item="Surface elevation",
        name="test",
        itemInfo=mikeio.EUMType.Significant_wave_height,
    )

    assert isinstance(test, protocols.ModelResult)
    assert isinstance(test, protocols.Extractable)

    dfsu_data = mikeio.open("tests/testdata/Oresund2D.dfsu")
    point_obs = PointObservation(
        "tests/testdata/SW/HKNA_Hm0.dfs0", item=0, x=4.2420, y=52.6887, name="HKNA"
    )
    track_obs = TrackObservation(
        "tests/testdata/SW/Alti_c2_Dutch.dfs0", item=3, name="C2"
    )

    test.extract(point_obs)
    test.extract(track_obs)

    c1 = test.extract_observation(point_obs)
    c2 = test.extract_observation(track_obs)
