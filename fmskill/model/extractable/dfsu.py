from typing import Union

import mikeio

from fmskill.model import protocols
from fmskill.model._base import ModelResultBase
from fmskill.observation import PointObservation, TrackObservation


class DfsuModelResult(ModelResultBase):
    def extract(self, observation: Union[PointObservation, TrackObservation]):
        pass


if __name__ == "__main__":
    dfsu = mikeio.open("tests/testdata/Oresund2D.dfsu")
    test = DfsuModelResult(dfsu, "test", "test", "test")

    assert isinstance(test, protocols.ModelResult)
    assert isinstance(test, protocols.Extractable)
