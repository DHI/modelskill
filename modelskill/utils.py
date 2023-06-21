from typing import List, Tuple
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from collections.abc import Iterable

POS_COORDINATE_NAME_MAPPING = {
    "lon": "x",
    "longitude": "x",
    "lat": "y",
    "latitude": "y",
    "east": "x",
    "north": "y",
    "x": "x",
    "y": "y",
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


def get_item_name_and_idx(item_names: List[str], item) -> Tuple[str, int]:
    """Returns the name and index of the requested variable, provided
    either as either a str or int."""
    n_items = len(item_names)
    if item is None:
        if n_items == 1:
            return item_names[0], 0
        else:
            raise ValueError(
                f"item must be specified when more than one item available. Available items: {item_names}"
            )
    if isinstance(item, int):
        if item < 0:  # Handle negative indices
            item = n_items + item
        if (item < 0) or (item >= n_items):
            raise IndexError(f"item {item} out of range (0, {n_items-1})")
        return item_names[item], item
    elif isinstance(item, str):
        if item not in item_names:
            raise KeyError(f"item must be one of {item_names}, got {item}.")
        return item, item_names.index(item)
    else:
        raise TypeError("item must be int or string")


def _parse_track_items(items, x_item, y_item, item):
    """If input has exactly 3 items we accept item=None"""
    if len(items) < 3:
        raise ValueError(
            f"Input has only {len(items)} items. It should have at least 3."
        )
    if item is None:
        if len(items) == 3:
            item = 2
        elif len(items) > 3:
            raise ValueError("Input has more than 3 items, but item was not given!")

    item, _ = get_item_name_and_idx(items, item)
    x_item, _ = get_item_name_and_idx(items, x_item)
    y_item, _ = get_item_name_and_idx(items, y_item)

    if (item == x_item) or (item == y_item) or (x_item == y_item):
        raise ValueError(
            f"x-item ({x_item}), y-item ({y_item}) and value-item ({item}) must be different!"
        )
    return [x_item, y_item, item]


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
