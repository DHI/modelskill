import pandas as pd

from fmskill.model import protocols
from fmskill.model._base import ModelResultBase
from fmskill.observation import PointObservation


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
        self.x = x
        self.y = y

    def compare(self, observation: PointObservation):
        pass


if __name__ == "__main__":
    test = PointModelResult(pd.DataFrame(), 1.0, 2.0, "test", "test", "test")

    assert isinstance(test, protocols.ModelResult)
    assert isinstance(test, protocols.Comparable)
    assert isinstance(test, protocols.PointObject)
    assert isinstance(test, protocols.PointModelResult)
