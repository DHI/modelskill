from pathlib import Path
from typing import Optional, Union

import mikeio
import pandas as pd
import xarray as xr

DfsType = Union[mikeio.Dfs0, mikeio.Dfsu]

ItemSpecifier = Optional[Union[int, str]]

DataInputType = Union[
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
