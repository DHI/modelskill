import warnings
import numpy as np
import pandas as pd
import xarray as xr
from collections.abc import Iterable

from fmskill import types

POS_COORDINATE_NAME_MAPPING = {
    "lon": "x",
    "longitude": "x",
    "lat": "y",
    "latitude": "y",
    "east": "x",
    "north": "y",
}
TIME_COORDINATE_NAME_MAPPING = {
    "t": "time",
    "date": "time",
}


def rename_coords_xr(ds: xr.Dataset) -> xr.Dataset:
    """Rename coordinates to standard names"""
    ds = ds.rename(
        {
            c: TIME_COORDINATE_NAME_MAPPING[c.lower()]
            for c in list(ds.coords) + list(ds.data_vars)
            if c.lower() in TIME_COORDINATE_NAME_MAPPING.keys()
        }
    )
    ds = ds.rename(
        {
            c: POS_COORDINATE_NAME_MAPPING[c.lower()]
            for c in list(ds.coords) + list(ds.data_vars)
            if c.lower() in POS_COORDINATE_NAME_MAPPING.keys()
        }
    )
    return ds


def rename_coords_pd(df: pd.DataFrame) -> pd.DataFrame:
    """Rename coordinates to standard names"""
    _mapping = {
        c: TIME_COORDINATE_NAME_MAPPING[c.lower()]
        for c in df.columns
        if c.lower() in TIME_COORDINATE_NAME_MAPPING.keys()
    }
    _mapping.update(
        {
            c: POS_COORDINATE_NAME_MAPPING[c.lower()]
            for c in df.columns
            if c.lower() in POS_COORDINATE_NAME_MAPPING.keys()
        }
    )
    return df.rename(columns=_mapping)


def get_item_name_dfs(dfs: types.DfsType, item: str) -> str:
    item_names = [i.name for i in dfs.items]
    n_items = len(item_names)
    if item is None:
        if n_items == 1:
            return item_names[0]
        else:
            raise ValueError(
                f"item must be specified when more than one item available. Available items: {item_names}"
            )
    elif isinstance(item, str):
        if item not in item_names:
            raise KeyError(f"item must be one of {item_names}")
        return item
    else:
        raise TypeError("item must be int or string")


def is_iterable_not_str(obj):
    """Check if an object is an iterable but not a string."""
    if isinstance(obj, str):
        return False
    if isinstance(obj, Iterable):
        return True
    return False


def make_unique_index(df_index, offset_duplicates=0.001, warn=True):
    """Given a non-unique DatetimeIndex, create a unique index by adding
    milliseconds to duplicate entries

    Parameters
    ----------
    df_index : DatetimeIndex
        non-unique temporal index
    offset_in_seconds : float, optional
        add this many seconds to consecutive duplicate entries, by default 0.01
    warn : bool, optional
        issue user warning?, by default True

    Returns
    -------
    DatetimeIndex
        Unique index
    """
    assert isinstance(df_index, pd.DatetimeIndex)
    if df_index.is_unique:
        return df_index
    if warn:
        warnings.warn(
            "Time axis has duplicate entries. Now adding milliseconds to non-unique entries to make index unique."
        )
    values = df_index.duplicated(keep=False).astype(float)  # keep='first'
    values[values == 0] = np.NaN

    missings = np.isnan(values)
    cumsum = np.cumsum(~missings)
    diff = np.diff(np.concatenate(([0.0], cumsum[missings])))
    values[missings] = -diff

    # np.median(np.diff(df.index.values))/100
    offset_in_ns = offset_duplicates * 1e9
    tmp = np.cumsum(values.astype(int)).astype("timedelta64[ns]")
    new_index = df_index + offset_in_ns * tmp
    return new_index
