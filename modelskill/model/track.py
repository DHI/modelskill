from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import get_args, Optional, List
import pandas as pd

import mikeio

from ._base import Quantity
from ..types import TrackType
from ..timeseries import TimeSeries

from ..utils import make_unique_index, get_item_name_and_idx


@dataclass
class TrackItem:
    x: str
    y: str
    values: str

    @property
    def all(self) -> List[str]:
        return [self.x, self.y, self.values]


class TrackModelResult(TimeSeries):
    """Construct a TrackModelResult from a dfs0 file,
    mikeio.Dataset or pandas.DataFrame

    Parameters
    ----------
    data : types.TrackType
        the input data or file path
    name : Optional[str], optional
        The name of the model result,
        by default None (will be set to file name or item name)
    item : str | int | None, optional
        If multiple items/arrays are present in the input an item
        must be given (as either an index or a string), by default None
    x_item : str | int | None, optional
        Item of the first coordinate of positions, by default None
    y_item : str | int | None, optional
        Item of the second coordinate of positions, by default None
    quantity : Optional[str], optional
        A string to identify the quantity, by default None
    """

    def __init__(
        self,
        data: TrackType,
        *,
        name: Optional[str] = None,
        item: str | int | None = None,
        quantity: Optional[Quantity] = None,
        x_item: str | int = 0,
        y_item: str | int = 1,
    ) -> None:
        assert isinstance(
            data, get_args(TrackType)
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

        ti = self._parse_track_items(item_names, x_item, y_item, item)
        name = name or ti.values

        # select relevant items and convert to dataframe
        assert isinstance(data, (mikeio.Dataset, pd.DataFrame))
        data = data[ti.all]
        if isinstance(data, mikeio.Dataset):
            df = data.to_dataframe()
        else:
            df = data

        df = df.rename(columns={ti.x: "x", ti.y: "y"})
        df.index = make_unique_index(df.index, offset_duplicates=0.001)

        # TODO move default quantity to TimeSeries?
        model_quantity = Quantity.undefined() if quantity is None else quantity
        super().__init__(data=df, name=name, quantity=model_quantity)

    @staticmethod
    def _parse_track_items(items, x_item, y_item, item) -> TrackItem:
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
        # return x_item, y_item, item
        return TrackItem(x=x_item, y=y_item, values=item)
