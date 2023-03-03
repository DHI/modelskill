from typing import Union
import pandas as pd
from fmskill.comparison import PointComparer, SingleObsComparer

from fmskill.model import protocols
from fmskill.model._base import ModelResultBase
from fmskill.observation import PointObservation, TrackObservation


class PointModelResult(ModelResultBase):
    def __init__(
        self,
        data: pd.DataFrame,
        x: float,
        y: float,
        item: str = None,
        itemInfo=None,
        name: str = None,
        quantity: str = None,
    ) -> None:
        super().__init__(data, item, itemInfo, name, quantity)
        assert (x is not None) and (
            y is not None
        ), "x and y must be specified when creating a PointModelResult."
        self.x = x
        self.y = y

    def extract_observation(
        self, observation: Union[PointObservation, TrackObservation], validate=True
    ) -> SingleObsComparer:
        super().extract_observation(observation, validate)

        if not isinstance(observation, PointObservation):
            raise ValueError(
                "Can only extract PointObservation from a PointModelResult."
            )
        return PointComparer(observation, self.data)


if __name__ == "__main__":
    test = PointModelResult(pd.DataFrame(), 1.0, 2.0, "test", "test", "test")

    assert isinstance(test, protocols.ModelResult)
    assert isinstance(test, protocols.Comparable)
    assert isinstance(test, protocols.PointObject)
    assert isinstance(test, protocols.PointModelResult)
