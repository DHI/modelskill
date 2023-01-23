from pathlib import Path
from typing import Union, Optional

import mikeio
import pandas as pd
import xarray as xr

from ..utils import _as_path
from .dfs import DfsModelResultItem, dfs_get_item_index
from .pandas import DataFramePointModelResult, DataFrameTrackModelResult
from .xarray import XArrayModelResultItem


ModelResultDataInput = Union[
    str,
    Path,
    list[str],
    list[Path],
    mikeio.DataArray,
    pd.DataFrame,
    pd.Series,
    xr.Dataset,
    xr.DataArray,
]

DfsType = Union[mikeio.Dfs0, mikeio.Dfsu]

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
        self._validate_input_data(data, item)

        # TODO: handle dataframe inputs, check if we could simply
        # convert pandas stuff to xarray
        input_data_handler_mapping = {
            str: ModelResult.from_filepath,
            Path: ModelResult.from_filepath,
            list: ModelResult.from_filepath,
            xr.Dataset: ModelResult.from_xarray_dataset,
            xr.DataArray: ModelResult.from_xarray_array,
            mikeio.DataArray: ModelResult.from_mikeio_array,
            pd.Series: ModelResult.from_pd_series,
        }

        return input_data_handler_mapping[type(data)](
            data,
            item,
            *args,
            **kwargs,
        )

        # if isinstance(data, (pd.DataFrame, pd.Series)):
        #     type_ = kwargs.pop("type", "point")
        #     if type_ == "point":
        #         mr = DataFramePointModelResult(data, *args, **kwargs)
        #     elif type_ == "track":
        #         mr = DataFrameTrackModelResult(data, *args, **kwargs)
        #     else:
        #         raise ValueError(f"type '{type_}' unknown (point, track)")
        #     return self._mr_or_mr_item(mr)

    @classmethod
    def from_filepath(
        cls,
        filepath: Union[str, Path, list],
        item: ItemSpecifier = None,
        *args,
        **kwargs,
    ):
        filename = _as_path(filepath)
        ext = filename.suffix
        if "dfs" in ext:
            dfs: DfsType = mikeio.open(filename)
            return ModelResult.from_dfs(dfs, item)
        else:
            if "*" not in str(filename):
                return ModelResult.from_xarray_dataset(
                    xr.open_dataset(filename), item, *args, **kwargs
                )
            elif isinstance(filepath, str) or isinstance(filepath, list):
                return ModelResult.from_xarray_dataset(
                    xr.open_mfdataset(filepath), item, *args, **kwargs
                )

    @classmethod
    def from_dfs(cls, dfs: DfsType, item: ItemSpecifier = None):
        """Create a ModelResult from a dfs instance"""
        idx = dfs_get_item_index(dfs, item)
        return DfsModelResultItem(
            dfs=dfs, itemInfo=dfs.items[idx], filename=None, item_index=idx
        )

    @classmethod
    def from_xarray_array(cls, da: xr.DataArray, *args, **kwargs):
        """
        Create a ModelResult from an xarray DataArray. This should be the interface
        used for creating single item ModelResults from any xarray data structure.
        """
        new_da = validate_and_format_xarray(da)
        return XArrayModelResultItem(new_da, *args, **kwargs)

    @classmethod
    def from_xarray_dataset(
        cls, ds: xr.Dataset, item: ItemSpecifier = None, *args, **kwargs
    ):
        """Create a ModelResult from an xarray Dataset. An item needs to be passed if
        the Dataset contains more than one item."""

        item = cls._xarray_get_item_name(ds, item)
        return cls.from_xarray_array(ds[item], *args, **kwargs)

    @classmethod
    def from_mikeio_array(cls, da: mikeio.DataArray, *args, **kwargs):
        """Create a ModelResult from a mikeio DataArray."""
        return cls.from_xarray_array(da.to_xarray(), *args, **kwargs)

    @classmethod
    def from_pd_series(cls, series: pd.Series, *args, **kwargs):
        """Create a ModelResult from a pandas Series."""
        return cls.from_xarray_array(series.to_xarray(), *args, **kwargs)

    @classmethod
    def from_pd_dataframe(
        cls, df: pd.DataFrame, item: ItemSpecifier = None, *args, **kwargs
    ):
        """Create a ModelResult from a pandas DataFrame. An item needs to be passed if
        the DataFrame contains more than one series."""
