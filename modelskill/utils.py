from __future__ import annotations
from typing import Sequence
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from collections.abc import Iterable

_RESERVED_NAMES = ["Observation", "time", "x", "y", "z"]

POS_COORDINATE_NAME_MAPPING = {
    "lon": "x",
    "longitude": "x",
    "lat": "y",
    "latitude": "y",
    "east": "x",
    "north": "y",
    "x": "x",
    "y": "y",
    "z": "z",
    "depth": "z",
}
TIME_COORDINATE_NAME_MAPPING = {
    "t": "time",
    "date": "time",
}


def rename_coords_xr(ds: xr.Dataset) -> xr.Dataset:
    """Rename coordinates to standard names"""
    var_names = [str(name).lower() for name in ds.variables]

    ds = ds.rename(
        {
            c: TIME_COORDINATE_NAME_MAPPING[c]
            for c in var_names
            if c in TIME_COORDINATE_NAME_MAPPING
        }
    )
    ds = ds.rename(
        {
            c: POS_COORDINATE_NAME_MAPPING[c]
            for c in var_names
            if c in POS_COORDINATE_NAME_MAPPING
        }
    )
    return ds


def rename_coords_pd(df: pd.DataFrame) -> pd.DataFrame:
    """Rename coordinates to standard names"""

    col_names = [str(name).lower() for name in df.columns]

    mapping = {
        c: TIME_COORDINATE_NAME_MAPPING[c]
        for c in col_names
        if c in TIME_COORDINATE_NAME_MAPPING.keys()
    }
    mapping.update(
        {
            c: POS_COORDINATE_NAME_MAPPING[c]
            for c in col_names
            if c in POS_COORDINATE_NAME_MAPPING.keys()
        }
    )
    return df.rename(columns=mapping)


# def get_item_name_and_idx(
#     item_names: List[str], item: int | str | None = None
# ) -> Tuple[str, int]:
#     """Returns the name and index of the requested variable, provided
#     either as either a str or int.

#     Examples
#     --------
#     >>> get_item_name_and_idx(['a', 'b', 'c'], 1)
#     ('b', 1)
#     >>> get_item_name_and_idx(['a', 'b', 'c'], 'a')
#     ('a', 0)
#     >>> get_item_name_and_idx(['a', 'b', 'c'], -1)
#     ('c', 2)
#     """
#     n_items = len(item_names)
#     if item is None:
#         if n_items == 1:
#             return item_names[0], 0
#         else:
#             raise ValueError(
#                 f"item must be specified when more than one item available. Available items: {item_names}"
#             )
#     if isinstance(item, int):
#         if item < 0:  # Handle negative indices
#             item = n_items + item
#         if (item < 0) or (item >= n_items):
#             raise IndexError(f"item {item} out of range (0, {n_items-1})")
#         return item_names[item], item
#     elif isinstance(item, str):
#         if item not in item_names:
#             raise KeyError(f"item must be one of {item_names}, got {item}.")
#         return item, item_names.index(item)
#     else:
#         raise TypeError("item must be int or string")


def is_iterable_not_str(obj):
    """Check if an object is an iterable but not a string."""
    if isinstance(obj, str):
        return False
    if isinstance(obj, Iterable):
        return True
    return False


def make_unique_index(
    df_index: pd.DatetimeIndex, offset_duplicates: float = 0.001, warn: bool = True
) -> pd.DatetimeIndex:
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
    values[values == 0] = np.nan

    missings = np.isnan(values)
    cumsum = np.cumsum(~missings)
    diff = np.diff(np.concatenate(([0.0], cumsum[missings])))
    values[missings] = -diff

    # np.median(np.diff(df.index.values))/100
    offset_in_ns = offset_duplicates * 1e9
    tmp = np.cumsum(values.astype(int)).astype("timedelta64[ns]")
    new_index = df_index + offset_in_ns * tmp
    assert isinstance(new_index, pd.DatetimeIndex)
    return new_index


def _get_name(x: int | str | None, valid_names: Sequence[str]) -> str:
    """Parse name/idx from list of valid names (e.g. obs from obs_names), return name"""
    return valid_names[_get_idx(x, valid_names)]


def _get_idx(x: int | str | None, valid_names: Sequence[str]) -> int:
    """Parse name/idx from list of valid names (e.g. obs from obs_names), return idx"""

    if x is None:
        if len(valid_names) == 1:
            return 0
        else:
            raise ValueError(
                f"Multiple items available. Must specify name or index. Available items: {valid_names}"
            )

    n = len(valid_names)
    if n == 0:
        raise ValueError(f"Cannot select {x} from empty list!")
    elif isinstance(x, str):
        if x in valid_names:
            idx = valid_names.index(x)
        else:
            raise KeyError(f"Name {x} could not be found in {valid_names}")
    elif isinstance(x, int):
        if x < 0:  # Handle negative indices
            x += n
        if x >= 0 and x < n:
            idx = x
        else:
            raise IndexError(f"Id {x} is out of range for {valid_names}")
    else:
        raise TypeError(f"Input {x} invalid! Must be None, str or int, not {type(x)}")
    return idx
