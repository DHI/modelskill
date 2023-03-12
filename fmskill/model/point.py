from pathlib import Path
from typing import get_args
import mikeio
import pandas as pd

from fmskill import types, utils
from fmskill.model._base import ModelResultBase


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

        if isinstance(data, (str, Path)):
            assert Path(data).suffix == ".dfs0", "File must be a dfs0 file"
            data = mikeio.read(data, items=item).to_dataframe()
        elif isinstance(data, mikeio.Dfs0):
            data = data.read(items=item).to_dataframe()
        elif isinstance(data, mikeio.Dataset):
            data = data.to_dataframe() if item is None else data[item].to_dataframe()
        elif isinstance(data, mikeio.DataArray):
            data = data.to_dataframe()
        elif isinstance(data, pd.Series):
            data = pd.DataFrame(data)  # to_frame?

        assert isinstance(
            data, get_args(types.PointType)
        ), "Could not construct PointModelResult from provided data"
        if isinstance(data, (str, Path)):
            assert Path(data).suffix == ".dfs0", "File must be a dfs0 file"
            data = mikeio.open(data)

        if isinstance(data, mikeio.Dfs0):
            item, idx = utils.get_item_name_and_idx_dfs(data.items, item)
            if itemInfo is None:
                itemInfo = data.items[idx]
            data = data.read()
            data = mikeio.Dataset(data[item]).to_dataframe().dropna()

        # TODO: series

        if isinstance(data, pd.DataFrame):
            if data.empty or len(data.columns) == 0:
                raise ValueError("No data.")

            item, _ = utils.get_item_name_and_idx(list(data.columns), item)
            data = data[[item]]

            data.index = utils.make_unique_index(data.index, offset_duplicates=0.001)

        super().__init__(data, item, itemInfo, name, quantity)
        self.x = x
        self.y = y
