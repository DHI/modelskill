from pathlib import Path
from typing import Optional, Union, Callable

import mikeio
import xarray as xr
import pandas as pd

from .utils import _as_path
from fmskill import types


# functions for converting various input formats to xarray.DataArray


def array_from_dataset(
    ds: xr.Dataset, item: types.ItemSpecifier = None, *args, **kwargs
) -> xr.DataArray:
    """Get a DataArray from a Dataset"""
    item = _xarray_get_item_name(ds, item)
    da = ds[item]
    return da


def eager_array_from_filepath(
    filepath: Union[str, Path, list], item: types.ItemSpecifier = None
):
    """Get a DataArray from a filepath. Does not support dfs files."""
    filename = _as_path(filepath)
    ext = filename.suffix
    # if "dfs" in ext:
    # dfs: DfsType = mikeio.open(filename)
    # return array_from_dfs(dfs, item)
    if "dfs" not in ext:
        if "*" not in str(filename):
            return array_from_dataset(xr.open_dataset(filename), item)
        elif isinstance(filepath, str) or isinstance(filepath, list):
            return array_from_dataset(xr.open_mfdataset(filepath), item)


def lazy_array_from_filepath(
    filepath: Union[str, Path, list], item: types.ItemSpecifier = None
) -> types.DfsType:
    """
    Return a lazy loading object for a filepath.
    Currently supported formats: .dfs0, .dfsu
    """
    filename = _as_path(filepath)
    ext = filename.suffix
    if "dfs" in ext:
        return mikeio.open(filename)


def array_from_pd_series(series: pd.Series, *args, **kwargs) -> xr.DataArray:
    """Get a DataArray from a pandas Series."""
    return series.to_xarray()


def array_from_pd_dataframe(
    df: pd.DataFrame, item: types.ItemSpecifier = None, *args, **kwargs
) -> xr.DataArray:
    """Get a DataArray from a pandas DataFrame."""
    if item is None:
        if len(df.columns) == 1:
            item = df.columns[0]
        else:
            item = _pd_get_column_name(df, item)
    return df[item].to_xarray()


def get_eager_loader(
    data: types.DataInputType,
) -> Callable[[types.DataInputType, types.ItemSpecifier], xr.DataArray]:
    """Check if the provided data can be loaded eagerly. If so, return the loader function,
    otherwise return None."""

    if isinstance(data, (str, Path)):
        _data = _as_path(data)
        if "dfs" in _data.suffix:
            # dfs are loaded lazily, so return None here
            return None

    return eager_loading_types_mapping.get(type(data))


def get_lazy_loader(
    data: types.DataInputType,
) -> types.DfsType:
    return lazy_loading_types_mapping.get(type(data))


eager_loading_types_mapping = {
    xr.DataArray: lambda x: x,
    xr.Dataset: array_from_dataset,
    str: eager_array_from_filepath,
    Path: eager_array_from_filepath,
    list: eager_array_from_filepath,
    pd.Series: array_from_pd_series,
    pd.DataFrame: array_from_pd_dataframe,
}

lazy_loading_types_mapping = {
    str: lazy_array_from_filepath,
    Path: lazy_array_from_filepath,
}


def _pd_get_column_name(df: pd.DataFrame, item: types.ItemSpecifier = None) -> str:
    columns = list(df.columns)
    if isinstance(item, str):
        if item not in columns:
            raise KeyError(f"item must be one of {columns}.")
        return item
    elif isinstance(item, int):
        if item < 0:
            item = len(columns) + item
        if (item < 0) or (item >= len(columns)):
            raise ValueError(
                f"item must be between 0 and {len(columns)-1} (or {-len(columns)} and -1)"
            )
        return columns[item]
    else:
        raise TypeError("column must be given as int or str")


def _xarray_get_item_name(ds: xr.Dataset, item, item_names=None) -> str:
    if item_names is None:
        item_names = list(ds.data_vars)
    n_items = len(item_names)
    if item is None:
        if n_items == 1:
            return item_names[0]
        else:
            raise ValueError(
                f"item must be specified when more than one item available. Available items: {item_names}"
            )
    if isinstance(item, int):
        if item < 0:  # Handle negative indices
            item = n_items + item
        if (item < 0) or (item >= n_items):
            raise IndexError(f"item {item} out of range (0, {n_items-1})")
        item = item_names[item]
    elif isinstance(item, str):
        if item not in item_names:
            raise KeyError(f"item must be one of {item_names}")
    else:
        raise TypeError("item must be int or string")
    return item


def validate_and_format_xarray(da: xr.DataArray):
    new_names = {}
    coords = da.coords
    for coord in coords:
        c = coord.lower()
        if ("x" not in new_names) and (("lon" in c) or ("east" in c)):
            new_names[coord] = "x"
        elif ("y" not in new_names) and (("lat" in c) or ("north" in c)):
            new_names[coord] = "y"
        elif ("time" not in new_names) and (("time" in c) or ("date" in c)):
            new_names[coord] = "time"

    if len(new_names) > 0:
        da = da.rename(new_names)

    cnames = da.coords
    for c in ["x", "y", "time"]:
        if c not in cnames:
            raise ValueError(f"{c} not found in coords {cnames}")

    if not isinstance(coords["time"].to_index(), pd.DatetimeIndex):
        raise ValueError(f"Time coordinate is not equivalent to DatetimeIndex")

    return da


def _validate_input_data(data, item) -> None:
    """Validates the input data to ensure that a loader will available for the provided data."""

    if isinstance(data, mikeio.Dataset):
        raise ValueError("mikeio.Dataset not supported, but mikeio.DataArray is")

    if not isinstance(data, types.DataInputType):
        raise ValueError(
            "Input type not supported (str, Path, mikeio.DataArray, DataFrame, xr.DataArray)"
        )
    if not isinstance(item, types.ItemSpecifier):
        raise ValueError("Invalid type for item argument (int, str, None)")
