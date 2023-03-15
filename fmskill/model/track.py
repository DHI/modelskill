from pathlib import Path
from typing import Union
import pandas as pd
import mikeio
from fmskill import types, utils

from fmskill.model import protocols
from fmskill.model._base import ModelResultBase


class TrackModelResult(ModelResultBase):
    def __init__(
        self,
        data: types.DataInputType,
        *,
        name: str = None,
        item: Union[str, int] = None,
        itemInfo=None,
        quantity: str = None,
        x_item: Union[str, int] = 0,
        y_item: Union[str, int] = 1,
    ) -> None:
        if isinstance(data, (str, Path)):
            assert Path(data).suffix == ".dfs0", "File must be a dfs0 file"
            name = name or Path(data).stem
            data = mikeio.read(data)  # now mikeio.Dataset

        # parse items
        if isinstance(data, mikeio.Dataset):
            item_names = [i.name for i in data.items]
        elif isinstance(data, pd.DataFrame):
            item_names = list(data.columns)
        else:
            raise ValueError("Could not construct TrackModelResult from provided data")
        items = utils._parse_track_items(item_names, x_item, y_item, item)
        item = items[-1]
        item, idx = utils.get_item_name_and_idx(item_names, item)
        name = name or item
        if itemInfo is None and isinstance(data, mikeio.Dataset):
            itemInfo = data.items[idx]

        # select relevant items and convert to dataframe
        data = data[items]
        if isinstance(data, mikeio.Dataset):
            data = data.to_dataframe()

        # data = utils.rename_coords_pd(data)
        # assert (
        #     "x" in data.columns and "y" in data.columns
        # ), "Data must have x and y columns to construct a TrackModelResult."

        data = data.rename(columns={items[0]: "x", items[1]: "y"})
        data.index = utils.make_unique_index(data.index, offset_duplicates=0.001)

        super().__init__(
            data=data, name=name, item=item, itemInfo=itemInfo, quantity=quantity
        )
