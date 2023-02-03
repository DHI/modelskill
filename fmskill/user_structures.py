from typing import Optional, Union

from fmskill import types
from fmskill.data_container import DataContainer


class Observation:
    def __new__(
        self,
        data: types.DataInputType,
        item: types.ItemSpecifier = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        name: Optional[str] = None,
    ) -> DataContainer:

        return DataContainer(
            data=data,
            item=item,
            is_observation=True,
            x=x,
            y=y,
            name=name,
        )


class ModelResult:
    def __new__(
        self,
        data: types.DataInputType,
        item: types.ItemSpecifier = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        name: Optional[str] = None,
    ) -> DataContainer:

        return DataContainer(
            data=data,
            item=item,
            is_result=True,
            x=x,
            y=y,
            name=name,
        )


class Comparer:
    def __init__(
        self,
        observations: Union[Observation, list[Observation]],
        results: Union[ModelResult, list[ModelResult]],
    ) -> None:
        self.observations = observations
        self.results = results
