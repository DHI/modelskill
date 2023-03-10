from pathlib import Path
from typing import get_args
import mikeio
import pandas as pd

from fmskill import types, utils
from fmskill.comparison import PointComparer, SingleObsComparer
from fmskill.model import protocols
from fmskill.model._base import ModelResultBase
from fmskill.observation import PointObservation, TrackObservation


class PointModelResult(ModelResultBase):
    def __init__(
        self,
        data: types.PointType,
        x: float = None,
        y: float = None,
        item: str = None,
        itemInfo=None,
        name: str = None,
        quantity: str = None,
    ) -> None:

        assert isinstance(
            data, get_args(types.PointType)
        ), "Could not construct PointModelResult from provided data"
        if isinstance(data, (str, Path)):
            assert Path(data).suffix == ".dfs0", "File must be a dfs0 file"
            data = mikeio.open(data)

        if isinstance(data, mikeio.Dfs0):
            item, idx = utils.get_item_name_and_idx_dfs(data, item)
            itemInfo = data.items[idx].type
            data = data.read()
            data = mikeio.Dataset(data[item]).to_dataframe().dropna()

        if isinstance(data, pd.DataFrame):
            if data.empty or len(data.columns) == 0:
                raise ValueError("No data.")

            item, _ = utils.get_item_name_and_idx_pd(data, item)
            data = data[[item]]
            if itemInfo is None:
                itemInfo = mikeio.EUMType.Undefined

            data.index = utils.make_unique_index(data.index, offset_duplicates=0.001)

        super().__init__(data, item, itemInfo, name, quantity)
        self.x = x
        self.y = y


if __name__ == "__main__":
    test = PointModelResult(pd.DataFrame(), 1.0, 2.0, "test", "test", "test")

    assert isinstance(test, protocols.ModelResult)
    assert isinstance(test, protocols.Comparable)
    assert isinstance(test, protocols.PointObject)
    assert isinstance(test, protocols.PointModelResult)
