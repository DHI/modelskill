from typing import Union
from pathlib import Path

import mikeio
import pandas as pd
import xarray as xr

from .dfs import DataArrayModelResultItem, DfsModelResult
from .pandas import DataFramePointModelResult, DataFrameTrackModelResult
from .xarray import XArrayModelResult


class ModelResult:
    """
    ModelResult factory returning a specialized ModelResult object
    depending on the input.

    * dfs0 or dfsu file
    * pandas.DataFrame/Series
    * NetCDF/Grib

    Note
    ----
    If an input has more than one item and the desired item is not
    specified as argument on construction, then the item of the
    modelresult 'mr' **must** be specified by e.g. mr[0] or mr['item_B']
    before connecting to an observation.

    Examples
    --------
    >>> mr = ModelResult("Oresund2D.dfsu")
    >>> mr_item = mr["Surface elevation"]
    >>> mr = ModelResult("Oresund2D_points.dfs0", name="Oresund")
    >>> mr_item = mr[0]
    >>> mr_item = ModelResult("Oresund2D.dfsu", item=0)
    >>> mr_item = ModelResult("Oresund2D.dfsu", item="Surface elevation")

    >>> mr = ModelResult(df)
    >>> mr = mr["Water Level"]
    >>> mr_item = ModelResult(df, item="Water Level")
    >>> mr_item = ModelResult(df, item="Water Level", itemInfo=mikeio.EUMType.Water_Level)

    >>> mr = ModelResult("ThirdParty.nc")
    >>> mr = mr["WL"]
    >>> mr_item = ModelResult("ThirdParty.nc", item="WL")
    >>> mr_item = ModelResult("ThirdParty.nc", item="WL", itemInfo=mikeio.EUMType.Water_Level)
    """

    def __new__(
        self,
        data: Union[
            str, mikeio.DataArray, pd.DataFrame, pd.Series, xr.Dataset, xr.DataArray
        ],
        *args,
        **kwargs,
    ):

        if isinstance(data, str):
            filename = Path(data)
            ext = filename.suffix
            if "dfs" in ext:
                mr = DfsModelResult(str(filename), *args, **kwargs)
                return self._mr_or_mr_item(mr)
            else:
                mr = XArrayModelResult(str(filename), *args, **kwargs)
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
