from pathlib import Path
from typing import Union, get_args, Optional, Tuple
import pandas as pd

import mikeio

from ._base import Quantity, ModelResultBase
from .. import types

from ..utils import make_unique_index, get_item_name_and_idx


class TrackModelResult(ModelResultBase):
    """Construct a TrackModelResult from a dfs0 file,
    mikeio.Dataset or pandas.DataFrame

    Parameters
    ----------
    data : types.TrackType
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
        name: Optional[str] = None,
        item: Optional[Union[str, int]] = None,
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

        x_name, y_name, item_name = self._parse_track_items(
            item_names, x_item, y_item, item
        )
        sel_item_names = [x_name, y_name, item_name]
        name = name or item_name

        # select relevant items and convert to dataframe
        assert isinstance(data, (mikeio.Dataset, pd.DataFrame))
        data = data[sel_item_names]
        if isinstance(data, mikeio.Dataset):
            df = data.to_dataframe()
        else:
            df = data

        df = df.rename(columns={x_name: "x", y_name: "y"})
        df.index = make_unique_index(df.index, offset_duplicates=0.001)

        super().__init__(data=df, name=name, quantity=quantity)

    @staticmethod
    def _parse_track_items(items, x_item, y_item, item) -> Tuple[str, str, str]:
        """If input has exactly 3 items we accept item=None"""
        if len(items) < 3:
            raise ValueError(
                f"Input has only {len(items)} items. It should have at least 3."
            )
        if item is None:
            if len(items) == 3:
                item = 2
            elif len(items) > 3:
                raise ValueError("Input has more than 3 items, but item was not given!")

        item, _ = get_item_name_and_idx(items, item)
        x_item, _ = get_item_name_and_idx(items, x_item)
        y_item, _ = get_item_name_and_idx(items, y_item)

        if (item == x_item) or (item == y_item) or (x_item == y_item):
            raise ValueError(
                f"x-item ({x_item}), y-item ({y_item}) and value-item ({item}) must be different!"
            )
        return x_item, y_item, item
