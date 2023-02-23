import pandas as pd

from fmskill.model import protocols
from fmskill.model._base import ModelResultBase
from fmskill.observation import TrackObservation


class TrackModelResult(ModelResultBase):
    def compare(self, observation: TrackObservation):
        pass


if __name__ == "__main__":
    test = TrackModelResult(pd.DataFrame(), "test", "test", "test")

    assert isinstance(test, protocols.ModelResult)
    assert isinstance(test, protocols.Comparable)
