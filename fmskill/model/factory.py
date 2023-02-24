import os
from pathlib import Path
from typing import Literal, Optional
import pandas as pd
import xarray as xr

from fmskill import types, utils, model


import mikeio

from .dfs import DataArrayModelResultItem, DfsModelResultItem
from ._pandas import DataFramePointModelResultItem, DataFrameTrackModelResultItem
from ._xarray import XArrayModelResultItem


type_lookup = {
    "point": model.PointModelResult,
    "track": model.TrackModelResult,
    "unstructured": model.DfsuModelResult,
    "grid": model.GridModelResult,
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

        if file_ext in [".dfs2", ".dfsu"]:
            data = mikeio.open(data)
            item = utils.get_item_name_dfs(data, item)

        elif file_ext == ".nc":
            data = xr.open_dataset(data)

        if model_type is not None:
            return type_lookup[model_type](data, item, name, quantity)

        if file_ext == ".dfs2":
            return model.GridModelResult(data, item, name, quantity)
        else:
            return model.DfsuModelResult(data, item, name, quantity)


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
