from typing import Any, Protocol, Union, runtime_checkable

import pandas as pd

from modelskill import types


@runtime_checkable
class ModelResult(Protocol):
    data: Any
    name: str
    quantity: types.Quantity
    time: pd.DatetimeIndex


@runtime_checkable
class Observation(Protocol):
    data: pd.DataFrame
    name: str


@runtime_checkable
class PointObject(Protocol):

    x: float
    y: float


@runtime_checkable
class PointObservation(PointObject, Observation, Protocol):
    ...


@runtime_checkable
class Comparable(ModelResult, Protocol):

    data: pd.DataFrame

    def compare(self, observation: Union[PointObservation, Observation]):
        # assert isinstance(self.data, ExtractableType), "data is not extractable"
        ...


@runtime_checkable
class PointModelResult(PointObject, Comparable, Protocol):
    ...
