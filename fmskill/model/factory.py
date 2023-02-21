import os
import pandas as pd

from .dfs import DfsModelResult, DataArrayModelResultItem, DfsModelResultItem
from .pandas import DataFramePointModelResult, DataFrameTrackModelResult
from .xarray import XArrayModelResult


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
                mr = XArrayModelResult(filename, *args, **kwargs)
                return self._mr_or_mr_item(mr)

        elif isinstance(data, mikeio.DataArray):
            return DataArrayModelResultItem(data, *args, **kwargs)
        elif isinstance(data, mikeio.Dataset):
            raise ValueError("mikeio.Dataset not supported, but mikeio.DataArray is")
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            type = kwargs.pop("type", "point")
            if type == "point":
                mr = DataFramePointModelResult(data, *args, **kwargs)
            elif type == "track":
                mr = DataFrameTrackModelResult(data, *args, **kwargs)
            else:
                raise ValueError(f"type '{type}' unknown (point, track)")
            return self._mr_or_mr_item(mr)
        elif isinstance(data, (xr.Dataset, xr.DataArray)):
            mr = XArrayModelResult(data, *args, **kwargs)
            return self._mr_or_mr_item(mr)
        else:
            raise ValueError(
                "Input type not supported (filename, mikeio.DataArray, DataFrame, xr.DataArray)"
            )

    @staticmethod
    def _mr_or_mr_item(mr):
        if mr._selected_item is not None:
            return mr[mr._selected_item]
        else:
            return mr
