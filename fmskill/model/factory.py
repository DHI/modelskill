import os
import pandas as pd
import xarray as xr

from .dfs import DfsModelResult
from .pandas import DataFrameModelResult


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
                if mr._selected_item is not None:
                    return mr[mr._selected_item]
                else:
                    return mr
            else:
                # return XrModelResult(filename, *args, **kwargs)
                raise NotImplementedError()
        elif isinstance(input, (pd.DataFrame, pd.Series)):
            mr = DataFrameModelResult(input, *args, **kwargs)
            if mr._selected_item is not None:
                return mr[mr._selected_item]
            else:
                return mr
        elif isinstance(input, (xr.Dataset, xr.DataArray)):
            raise NotImplementedError()
            # return XrModelResult(input, *args, **kwargs)
        else:
            raise ValueError("Input type not supported (filename or DataFrame)")


# class ModelResult(ModelResultInterface):


#     @property
#     def itemInfo(self):
#         if self.item is None:
#             return eum.ItemInfo(eum.EUMType.Undefined)
#         else:
#             # if isinstance(self.item, str):
#             self.item = self._parse_item(self.item)
#             return self.dfs.items[self.item]


#     def _get_model_item(self, item, mod_items=None) -> eum.ItemInfo:
#         """Given str or int find corresponding model itemInfo"""
#         if mod_items is None:
#             mod_items = self.dfs.items
#         n_items = len(mod_items)
#         if isinstance(item, int):
#             if (item < 0) or (item >= n_items):
#                 raise ValueError(f"item number must be between 0 and {n_items}")
#         elif isinstance(item, str):
#             item_names = [i.name for i in mod_items]
#             try:
#                 item = item_names.index(item)
#             except ValueError:
#                 raise ValueError(f"item not found in model items ({item_names})")
#         else:
#             raise ValueError("item must be an integer or a string")
#         return mod_items[item]
