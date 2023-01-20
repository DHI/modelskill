from pathlib import Path
from typing import Union, Optional

import mikeio
import pandas as pd
import xarray as xr

from ..utils import _as_path
from .dfs import DataArrayModelResultItem, DfsModelResult, DfsModelResultItem
from .pandas import DataFramePointModelResult, DataFrameTrackModelResult
from .xarray import XArrayModelResult


ModelResultDataInput = Union[
    str,
    Path,
    mikeio.DataArray,
    pd.DataFrame,
    pd.Series,
    xr.Dataset,
    xr.DataArray,
]

ItemSpecifier = Optional[Union[int, str]]


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
        data: ModelResultDataInput,
        item: ItemSpecifier = None,
        *args,
        **kwargs,
    ):
        self._validate_result(data, item)

        if isinstance(data, (str, Path)):
            filename = _as_path(data)
            ext = filename.suffix
            if "dfs" in ext:
                dfs = mikeio.open(filename)
                info, item_index = self._validate_and_get_item_info(dfs, item)

                return DfsModelResultItem(
                    dfs=dfs, itemInfo=info, filename=filename, item_index=item_index
                )
            else:
                mr = XArrayModelResult(filename, *args, **kwargs)
                return self._mr_or_mr_item(mr)

        elif isinstance(data, mikeio.DataArray):
            return DataArrayModelResultItem(data, *args, **kwargs)
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

    @staticmethod
    def _validate_result(data, item) -> None:
        if isinstance(data, mikeio.Dataset):
            raise ValueError("mikeio.Dataset not supported, but mikeio.DataArray is")

        if not isinstance(data, ModelResultDataInput):
            raise ValueError(
                "Input type not supported (str, Path, mikeio.DataArray, DataFrame, xr.DataArray)"
            )
        if not isinstance(item, ItemSpecifier):
            raise ValueError("Invalid type for item argument (int, str, None)")

    @staticmethod
    def _validate_and_get_item_info(dfs, item):
        available_names = [i.name for i in dfs.items]
        lower_case_names = [i.lower() for i in available_names]
        if item is None:
            if len(dfs.items) > 1:
                raise ValueError(
                    f"Found more than one item in dfs. Please specify item. Available: {available_names}"
                )
            else:
                idx = 0
        elif isinstance(item, str):
            if item.lower() not in lower_case_names:
                raise ValueError(
                    f"Requested item {item} not found in dfs file. Available: {available_names}"
                )
            idx = lower_case_names.index(item.lower())

        elif isinstance(item, int):
            idx = item
            n_items = len(dfs.items)
            if idx < 0:  # Handle negative indices
                idx = n_items + idx
            if (idx < 0) or (idx >= n_items):
                raise IndexError(f"item {item} out of range (0, {n_items-1})")

        return dfs.items[idx], idx
