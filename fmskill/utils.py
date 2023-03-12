from typing import List, Tuple
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from collections.abc import Iterable
import mikeio

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


def get_item_name_and_idx_dfs(items: List[mikeio.ItemInfo], item) -> Tuple[str, int]:
    """Returns the name and index of the requested variable, provided
    either as either a str or int."""
    item_names = [i.name for i in items]
    return get_item_name_and_idx(item_names, item)


def get_item_name_and_idx_xr(ds: xr.Dataset, item) -> Tuple[str, int]:
    """Returns the name and index of the requested data variable, provided
    either as either a str or int."""
    item_names = list(ds.data_vars)
    return get_item_name_and_idx(item_names, item)


def get_item_name_and_idx_pd(df: pd.DataFrame, item) -> Tuple[str, int]:
    """Returns the name and index of the requested data variable, provided
    either as either a str or int."""
    item_names = list(df.columns)
    return get_item_name_and_idx(item_names, item)


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


def parse_itemInfo(itemInfo):
    if itemInfo is None:
        return mikeio.ItemInfo(mikeio.EUMType.Undefined)
    if isinstance(itemInfo, mikeio.ItemInfo):
        return itemInfo
    return mikeio.ItemInfo(itemInfo)


def validate_item_eum(mod_item: mikeio.ItemInfo, obs) -> bool:
    """Check that observation and model item eum match"""
    ok = True
    if obs.itemInfo.type == mikeio.EUMType.Undefined:
        warnings.warn(f"{obs.name}: Cannot validate as type is Undefined.")
        return ok

    if mod_item.type != obs.itemInfo.type:
        ok = False
        warnings.warn(
            f"{obs.name}: Item type should match. Model item: {mod_item.type.display_name}, obs item: {obs.itemInfo.type.display_name}"
        )
    if mod_item.unit != obs.itemInfo.unit:
        ok = False
        warnings.warn(
            f"{obs.name}: Unit should match. Model unit: {mod_item.unit.display_name}, obs unit: {obs.itemInfo.unit.display_name}"
        )
    return ok
