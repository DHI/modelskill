import os
import pandas as pd

from .dfs import DataArrayModelResultItem, DfsModelResultItem
from ._pandas import DataFramePointModelResultItem, DataFrameTrackModelResultItem
from ._xarray import XArrayModelResultItem


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
