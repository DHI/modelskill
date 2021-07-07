import os
import pandas as pd
import xarray as xr

from .dfs import DfsModelResult
from .pandas import DataFrameModelResult, DataFrameTrackModelResult
from .xarray import XArrayModelResult


class ModelResult:
    """
    ModelResult factory returning a specialized ModelResult object
    depending on the input.

    * dfs0 or dfsu file
    * pandas.DataFrame/Series
    * NetCDF/Grib: Under development!

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
    """

    def __new__(self, input, *args, **kwargs):
        if isinstance(input, str):
            filename = input
            ext = os.path.splitext(filename)[-1]
            if "dfs" in ext:
                mr = DfsModelResult(filename, *args, **kwargs)
                return self._mr_or_mr_item(mr)
            else:
                mr = XArrayModelResult(filename, *args, **kwargs)
                return self._mr_or_mr_item(mr)

        elif isinstance(input, (pd.DataFrame, pd.Series)):
            type = kwargs.pop("type", "point")
            if type == "point":
                mr = DataFrameModelResult(input, *args, **kwargs)
            elif type == "track":
                mr = DataFrameTrackModelResult(input, *args, **kwargs)
            else:
                raise ValueError(f"type '{type}' unknown (point, track)")
            return self._mr_or_mr_item(mr)
        elif isinstance(input, (xr.Dataset, xr.DataArray)):
            mr = XArrayModelResult(input, *args, **kwargs)
            return self._mr_or_mr_item(mr)
        else:
            raise ValueError("Input type not supported (filename or DataFrame)")

    @staticmethod
    def _mr_or_mr_item(mr):
        if mr._selected_item is not None:
            return mr[mr._selected_item]
        else:
            return mr
