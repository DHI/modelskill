from typing import Any
from pathlib import Path
from typing import Optional

import pandas as pd

# from ..quantity import Quantity

from ._base import SelectedItems
from ..obs import NetworkLocationObservation, PointObservation, TrackObservation
from .point import PointModelResult


class NetworkModelResult:
    def __init__(
        self,
        data: str | Path,
        *,
        name: str,
        item: str | int | None = None,
        # quantity: Optional[Quantity] = None,
        aux_items: Optional[list[int | str]] = None,
    ) -> None:
        import mikeio1d

        df = mikeio1d.Res1D(data).to_dataframe()

        self.data = df
        self.time = df.index
        self.name = name
        self.item = item

        # TODO load from file
        data_vars = ["Discharge", "Water Level"]

        sel_items = SelectedItems.parse(data_vars, item=item, aux_items=aux_items)

        self.sel_items = sel_items

    def extract(
        self,
        observation: PointObservation | TrackObservation | NetworkLocationObservation,
        **kwargs: Any,
    ) -> PointModelResult:
        if not isinstance(observation, NetworkLocationObservation):
            raise TypeError(
                "NetworkModelResult can only extract NetworkLocationObservation"
            )
        col = f"{self.item}:{observation.reach}:{observation.chainage}"

        df = pd.DataFrame()
        df[observation.name] = self.data[col]
        df.index = self.time
        return PointModelResult(
            data=df,
            name=self.name,
            item=observation.name,
        )
