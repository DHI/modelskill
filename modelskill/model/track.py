from pathlib import Path
from typing import Union, get_args, Optional
import pandas as pd

import mikeio

from ._base import Quantity, ModelResultBase
from .. import types, utils


class TrackModelResult(ModelResultBase):
    """Construct a TrackModelResult from a dfs0 file,
    mikeio.Dataset or pandas.DataFrame

    Parameters
    ----------
    data : types.UnstructuredType
        the input data or file path
    name : Optional[str], optional
        The name of the model result,
        by default None (will be set to file name or item name)
    item : Optional[Union[str, int]], optional
        If multiple items/arrays are present in the input an item
        must be given (as either an index or a string), by default None
    x_item : Optional[Union[str, int]], optional
        Item of the first coordinate of positions, by default None
    y_item : Optional[Union[str, int]], optional
        Item of the second coordinate of positions, by default None
    quantity : Optional[str], optional
        A string to identify the quantity, by default None
    """

    def __init__(
        self,
        data: types.TrackType,
        *,
        name: str = None,
        item: Union[str, int] = None,
        quantity: Optional[Quantity] = None,
        x_item: Union[str, int] = 0,
        y_item: Union[str, int] = 1,
    ) -> None:
        assert isinstance(
            data, get_args(types.TrackType)
        ), "Could not construct TrackModelResult from provided data."

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

        # select relevant items and convert to dataframe
        data = data[items]
        if isinstance(data, mikeio.Dataset):
            data = data.to_dataframe()

        data = data.rename(columns={items[0]: "x", items[1]: "y"})
        data.index = utils.make_unique_index(data.index, offset_duplicates=0.001)

        super().__init__(data=data, name=name, quantity=quantity)
