from pathlib import Path
import pandas as pd
import mikeio
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
        if isinstance(data, (str, Path)):
            assert Path(data).suffix == ".dfs0", "File must be a dfs0 file"
            data = mikeio.read(data, items=item).to_dataframe()
        elif isinstance(data, mikeio.Dfs0):
            data = data.read(items=item).to_dataframe()
        if isinstance(data, mikeio.Dataset):
            data = data.to_dataframe() if item is None else data[item].to_dataframe()
        if isinstance(data, mikeio.DataArray):
            data = data.to_dataframe()

        if isinstance(data, pd.DataFrame):

            data = utils.rename_coords_pd(data)
            assert (
                "x" in data.columns and "y" in data.columns
            ), "Data must have x and y columns to construct a TrackModelResult."

            item, _ = utils.get_item_name_and_idx(list(data.columns), item)
            data = data[["x", "y", item]]
            data.index = utils.make_unique_index(data.index, offset_duplicates=0.001)

        super().__init__(data, item, itemInfo, name, quantity, **kwargs)
