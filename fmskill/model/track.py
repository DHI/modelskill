from numpy import isin
import pandas as pd
from fmskill import types, utils

from fmskill.model import protocols
from fmskill.model._base import ModelResultBase


class TrackModelResult(ModelResultBase):
    def __init__(
        self,
        data: types.DataInputType,
        item: str = None,
        itemInfo=None,
        name: str = None,
        quantity: str = None,
        **kwargs
    ) -> None:
        if isinstance(data, pd.DataFrame):

            data = utils.rename_coords_pd(data)
            assert (
                "x" in data.columns and "y" in data.columns
            ), "Data must have x and y columns to construct a TrackModelResult."

            item, _ = utils.get_item_name_and_idx_pd(data, item)
            data = data[["x", "y", item]]
            data.index = utils.make_unique_index(data.index, offset_duplicates=0.001)

        super().__init__(data, item, itemInfo, name, quantity, **kwargs)


if __name__ == "__main__":
    test = TrackModelResult(pd.DataFrame(), "test", "test", "test")

    assert isinstance(test, protocols.ModelResult)
    assert isinstance(test, protocols.Comparable)
