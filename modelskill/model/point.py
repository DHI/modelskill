from pathlib import Path
from typing import Optional, Union, get_args
import mikeio
import pandas as pd

from .. import types, utils
from ..types import Quantity
from ..timeseries import TimeSeries  # TODO move to main module


class PointModelResult(TimeSeries):
    """Construct a PointModelResult from a dfs0 file,
    mikeio.Dataset/DataArray or pandas.DataFrame/Series

    Parameters
    ----------
    data : types.UnstructuredType
        the input data or file path
    name : Optional[str], optional
        The name of the model result,
        by default None (will be set to file name or item name)
    x : float, optional
        first coordinate of point position, by default None
    y : float, optional
        second coordinate of point position, by default None
    item : Optional[Union[str, int]], optional
        If multiple items/arrays are present in the input an item
        must be given (as either an index or a string), by default None
    quantity : Quantity, optional
        Model quantity, for MIKE files this is inferred from the EUM information
    """

    def __init__(
        self,
        data: types.PointType,
        *,
        name: Optional[str] = None,  # TODO should maybe be required?
        x: Optional[float] = None,
        y: Optional[float] = None,
        item: Optional[Union[str, int]] = None,
        quantity: Optional[Quantity] = None,
    ) -> None:
        assert isinstance(
            data, get_args(types.PointType)
        ), "Could not construct PointModelResult from provided data"

        if isinstance(data, (str, Path)):
            assert Path(data).suffix == ".dfs0", "File must be a dfs0 file"
            name = name or Path(data).stem
            data = mikeio.read(data)  # now mikeio.Dataset
        elif isinstance(data, mikeio.Dfs0):
            data = data.read()  # now mikeio.Dataset

        # parse item and convert to dataframe
        if isinstance(data, mikeio.Dataset):
            item_names = [i.name for i in data.items]
            item, idx = utils.get_item_name_and_idx(item_names, item)
            data = data[[item]].to_dataframe()
        elif isinstance(data, mikeio.DataArray):
            item = item or data.name
            data = mikeio.Dataset({data.name: data}).to_dataframe()
        elif isinstance(data, pd.DataFrame):
            item_names = list(data.columns)
            item, idx = utils.get_item_name_and_idx(item_names, item)
            data = data[[item]]
        elif isinstance(data, pd.Series):
            data = pd.DataFrame(data)  # to_frame?
            item = item or data.columns[0]
        else:
            raise ValueError("Could not construct PointModelResult from provided data")

        name = name or item

        # basic processing
        data = data.dropna()
        if data.empty or len(data.columns) == 0:
            raise ValueError("No data.")
        data.index = utils.make_unique_index(data.index, offset_duplicates=0.001)

        super().__init__(data=data, name=name, quantity=quantity)
        self.x = x
        self.y = y

        # self.gtype = GeometryType.POINT
