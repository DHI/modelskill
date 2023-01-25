"""
Problem example:
When loading a dfs0 file, this can happen either eagerly or lazily,
depending on whether the file is a result (lazy) or an observation (eager).
Furthermore, we want to determine if the data is point or track data.
If the data contains multiple unique coordinates (e.g. latitude/longitude, x/y),
it is track data. However, the coordinates may also be stored as data variables of
the DataSet. Therefore, we can't simply convert to a DataArray based only on the
provided item. 

Strategy:
1. Parse all eagerly loadable formats to a xarray.DataSet.
2. Check coords and data variables for coordinates to determine if point or track data.
   If track data is present, make sure it is part of the coords.
3. Using the specified item, reduce to a xarray.DataArray.

The end result should be a DataArray having either only a time coordinate (point data)
or time and geographical coordinates (track data).
The logic for creating the dataset should be separated from the logic for creating the DataArray.
"""

from pathlib import Path
from typing import Optional, Union, Callable, Tuple

import mikeio
import xarray as xr
import pandas as pd

from .utils import _as_path
from fmskill import types

POS_COORDINATE_NAME_MAPPING = {
    "x": "x",
    "y": "y",
    "lon": "x",
    "longitude": "x",
    "lat": "y",
    "latitude": "y",
    "east": "x",
    "north": "y",
}
TIME_COORDINATE_NAME_MAPPING = {"time": "time", "t": "time", "date": "time"}


# functions for converting various input formats to xarray.DataArray


# def array_from_dataset(
#     ds: xr.Dataset, item: types.ItemSpecifier = None, *args, **kwargs
# ) -> Tuple[xr.DataArray, str]:
#     """Get a DataArray from a Dataset"""
#     item = _xarray_get_item_name(ds, item)
#     da = ds[item]
#     return da, item


def eager_ds_from_filepath(filepath: Union[str, Path, list]):
    """Get a DataSet from a filepath. Does not support dfs files."""
    filename = _as_path(filepath)
    ext = filename.suffix
    if "dfs" not in ext:
        if "*" not in str(filename):
            return xr.open_dataset(filename)
        elif isinstance(filepath, str) or isinstance(filepath, list):
            return xr.open_mfdataset(filepath)


def lazy_ds_from_filepath(filepath: Union[str, Path, list]) -> types.DfsType:
    """
    Return a lazy loading object for a filepath.
    Currently supported formats: .dfs0, .dfsu
    """
    filename = _as_path(filepath)
    ext = filename.suffix
    if "dfs" in ext:
        dfs = mikeio.open(filename)
        # item = _dfs_get_item_index(dfs, item)
        return dfs


# def array_from_pd_series(
#     series: pd.Series, *args, **kwargs
# ) -> Tuple[xr.DataArray, str]:
#     """Get a DataArray from a pandas Series."""
#     return series.to_xarray(), series.name


def array_from_pd_dataframe(df: pd.DataFrame, *args, **kwargs) -> xr.Dataset:
    """Get a DataSet from a pandas DataFrame."""
    return df.to_xarray()


def get_eager_loader(
    data: types.DataInputType,
) -> Callable[[types.DataInputType], xr.Dataset]:
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
    xr.DataArray: lambda x: x.to_dataset(),
    xr.Dataset: lambda x: x,
    str: eager_ds_from_filepath,
    Path: eager_ds_from_filepath,
    list: eager_ds_from_filepath,
    # pd.Series: array_from_pd_series,
    pd.DataFrame: array_from_pd_dataframe,
}

lazy_loading_types_mapping = {
    str: lazy_ds_from_filepath,
    Path: lazy_ds_from_filepath,
}


def parse_ds_coords(ds: xr.Dataset) -> xr.Dataset:
    """Parse the coordinates of a dataset. If track data is present, make sure it is part of the coords."""

    ds = _ensure_dims(ds)
    ds = _rename_coords(ds)

    if "time" not in ds.coords:
        raise ValueError("time coordinate not found")

    if not isinstance(ds.coords["time"].to_index(), pd.DatetimeIndex):
        raise ValueError(f"Time coordinate is not equivalent to DatetimeIndex")

    # check if track data is already part of the coords
    if all(d in ds.coords for d in ("x", "y", "time")):
        return ds

    # check if track data is stored as data variables
    data_vars = [c.lower() for c in ds.data_vars]
    if any(c in data_vars for c in POS_COORDINATE_NAME_MAPPING.keys()):
        ds = ds.set_coords(
            [c for c in ds.data_vars if c.lower() in POS_COORDINATE_NAME_MAPPING.keys()]
        )
        return _rename_coords(ds)

    # if no track data is present, we are dealing with point data
    # just rename the time coordinate and return
    return ds


def _ensure_dims(ds: xr.Dataset) -> xr.Dataset:
    def _already_correct(name: str):
        return name.lower() in ("time", "x", "y")

    for d in ds.dims:
        if d.lower() in POS_COORDINATE_NAME_MAPPING.keys() and not _already_correct(d):
            ds = ds.rename_dims({d: POS_COORDINATE_NAME_MAPPING[d.lower()]})
        if d.lower() in TIME_COORDINATE_NAME_MAPPING.keys() and not _already_correct(d):
            ds = ds.rename_dims({d: TIME_COORDINATE_NAME_MAPPING[d.lower()]})

    for d in ["time", "x", "y"]:
        if d not in ds.dims:
            ds = ds.expand_dims(d)

    return ds


def _rename_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename coordinates to standard names"""
    ds = ds.rename(
        {
            c: TIME_COORDINATE_NAME_MAPPING[c.lower()]
            for c in ds.coords
            if c.lower() in TIME_COORDINATE_NAME_MAPPING.keys()
        }
    )
    ds = ds.rename(
        {
            c: POS_COORDINATE_NAME_MAPPING[c.lower()]
            for c in ds.coords
            if c.lower() in POS_COORDINATE_NAME_MAPPING.keys()
        }
    )
    return ds


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


def xarray_get_item_name(ds: xr.Dataset, item, item_names=None) -> str:
    """Returns the name of the requested data variable, provided either as either a str or int."""
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


def _dfs_get_item_index(dfs, item):
    available_names = [i.name for i in dfs.items]
    lower_case_names = [i.lower() for i in available_names]
    if item is None:
        if len(dfs.items) > 1:
            raise ValueError(
                f"Found more than one item in dfs. Please specify item. Available: {available_names}"
            )
        else:
            return 0
    elif isinstance(item, str):
        if item.lower() not in lower_case_names:
            raise ValueError(
                f"Requested item {item} not found in dfs file. Available: {available_names}"
            )
        return lower_case_names.index(item.lower())

    elif isinstance(item, int):
        idx = item
        n_items = len(dfs.items)
        if idx < 0:  # Handle negative indices
            idx = n_items + idx
        if (idx < 0) or (idx >= n_items):
            raise IndexError(f"item {item} out of range (0, {n_items-1})")

        return idx


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

    cnames = list(da.coords)
    for c in ["x", "y", "time"]:
        if c not in cnames:
            raise ValueError(f"{c} not found in coords {cnames}")

    if not isinstance(coords["time"].to_index(), pd.DatetimeIndex):
        raise ValueError(f"Time coordinate is not equivalent to DatetimeIndex")

    return da


def validate_input_data(data, item) -> None:
    """Validates the input data to ensure that a loader will available for the provided data."""

    if isinstance(data, mikeio.Dataset):
        raise ValueError("mikeio.Dataset not supported, but mikeio.DataArray is")

    if not isinstance(data, types.DataInputType):
        raise ValueError(
            "Input type not supported (str, Path, mikeio.DataArray, DataFrame, xr.DataArray)"
        )
    if not isinstance(item, types.ItemSpecifier):
        raise ValueError("Invalid type for item argument (int, str, None)")
