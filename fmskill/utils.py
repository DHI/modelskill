import warnings
import numpy as np
import pandas as pd
from collections.abc import Iterable


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
