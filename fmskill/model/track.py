from typing import Union
import warnings
import pandas as pd
from fmskill.comparison import SingleObsComparer, TrackComparer

from fmskill.model import protocols
from fmskill.model._base import ModelResultBase
from fmskill.observation import PointObservation, TrackObservation


class TrackModelResult(ModelResultBase):
    def extract_observation(
        self, observation: Union[PointObservation, TrackObservation], validate=True
    ) -> SingleObsComparer:
        super().extract_observation(observation, validate)

        if not isinstance(observation, TrackObservation):
            raise ValueError(
                "Can only extract TrackObservation from a TrackModelResult."
            )
        comparer = TrackComparer(observation, self.data)
        if len(comparer.data) == 0:
            warnings.warn(f"No overlapping data in found for obs '{observation.name}'!")
            comparer = None

        return comparer


if __name__ == "__main__":
    test = TrackModelResult(pd.DataFrame(), "test", "test", "test")

    assert isinstance(test, protocols.ModelResult)
    assert isinstance(test, protocols.Comparable)
