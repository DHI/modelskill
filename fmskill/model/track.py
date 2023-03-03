from typing import Union
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
        return TrackComparer(observation, self.data)


if __name__ == "__main__":
    test = TrackModelResult(pd.DataFrame(), "test", "test", "test")

    assert isinstance(test, protocols.ModelResult)
    assert isinstance(test, protocols.Comparable)
