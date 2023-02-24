import os
from pathlib import Path
from typing import Literal, Optional
import pandas as pd
import xarray as xr

from fmskill import types, utils, model


import mikeio

from .legacy_dfs import DataArrayModelResultItem, DfsModelResultItem
from .legacy_pandas import DataFramePointModelResultItem, DataFrameTrackModelResultItem
from .legacy_xarray import XArrayModelResultItem


from enum import Enum, auto


class ModelType(Enum):
    Point = (auto(),)
    Track = (auto(),)
    Unstructured = (auto(),)
    Grid = auto()

    def __str__(self) -> str:
        return self.name.lower()

    def __eq__(self, __o: object) -> bool:
        return str(self) == str(__o)

    def from_string(s: str) -> "ModelType":
        try:
            return ModelType[s.capitalize()]
        except KeyError as e:
            raise KeyError(
                f"ModelType {s} not recognized. Available options: {[m.name for m in ModelType]}"
            ) from e


type_lookup = {
    ModelType.Point: model.PointModelResult,
    ModelType.Track: model.TrackModelResult,
    ModelType.Unstructured: model.DfsuModelResult,
    ModelType.Grid: model.GridModelResult,
}


class ModelResult_new:
    def __new__(
        cls,
        data: types.DataInputType,
        model_type: Optional[Literal["point", "track", "unstructured", "grid"]] = None,
        item: Optional[str] = None,
        name: Optional[str] = None,
        quantity: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ):
        # WIP!

        if isinstance(data, (str, Path)):
            data = Path(data)
            file_ext = data.suffix.lower()
        else:
            file_ext = None

        if file_ext in [".dfs2", ".dfsu", ".dfs0"]:
            data = mikeio.open(data)
            item = utils.get_item_name_dfs(data, item)

        elif file_ext == ".nc":
            data = xr.open_dataset(data)

        if model_type is not None:
            model_type = ModelType.from_string(model_type)
            return type_lookup[model_type](data, item, name, quantity)

        if file_ext == ".dfs2":
            return model.GridModelResult(data, item, name, quantity)
        elif file_ext == ".dfsu":
            return model.DfsuModelResult(data, item, name, quantity)
        elif file_ext == ".dfs0":



class ModelResult:
    """
    ModelResult factory returning a specialized ModelResult object
    depending on the data input.

    * dfs0 or dfsu file
    * pandas.DataFrame/Series
    * NetCDF/Grib

    Note
    ----
    If a data input has more than one item, the desired item **must** be
    specified as argument on construction.

    Examples
    --------
    >>> mr = ModelResult("Oresund2D.dfsu", item=0)
    >>> mr = ModelResult("Oresund2D.dfsu", item="Surface elevation")

    >>> mr = ModelResult(df, item="Water Level")
    >>> mr = ModelResult(df, item="Water Level", itemInfo=mikeio.EUMType.Water_Level)

    >>> mr = ModelResult("ThirdParty.nc", item="WL")
    >>> mr = ModelResult("ThirdParty.nc", item="WL", itemInfo=mikeio.EUMType.Water_Level)
    """

    def __new__(self, data, *args, **kwargs):
        import xarray as xr
        import mikeio

        if isinstance(data, str):
            filename = data
            ext = os.path.splitext(filename)[-1]
            if "dfs" in ext:
                return DfsModelResultItem(filename, *args, **kwargs)
            else:
                return XArrayModelResultItem(filename, *args, **kwargs)

        elif isinstance(data, mikeio.DataArray):
            return DataArrayModelResultItem(data, *args, **kwargs)
        elif isinstance(data, mikeio.Dataset):
            raise ValueError("mikeio.Dataset not supported, but mikeio.DataArray is")
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            type = kwargs.pop("type", "point")
            if type == "point":
                return DataFramePointModelResultItem(data, *args, **kwargs)
            elif type == "track":
                return DataFrameTrackModelResultItem(data, *args, **kwargs)
            else:
                raise ValueError(f"type '{type}' unknown (point, track)")

        elif isinstance(data, (xr.Dataset, xr.DataArray)):
            return XArrayModelResultItem(data, *args, **kwargs)
        else:
            raise ValueError(
                "Input type not supported (filename, mikeio.DataArray, DataFrame, xr.DataArray)"
            )
