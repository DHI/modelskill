from pathlib import Path
from typing import Union

import mikeio
import pandas as pd
import xarray as xr

DataInputType = Union[
    str,
    Path,
    list[str],
    list[Path],
    mikeio.DataArray,
    mikeio.Dataset,
    pd.DataFrame,
    pd.Series,
    xr.Dataset,
    xr.DataArray,
]

ExtractableType = Union[
    mikeio.dfsu._Dfsu,
    mikeio.Dfs2,
    mikeio.DataArray,
    xr.DataArray,
    xr.Dataset,
]

DfsType = Union[mikeio.Dfs0, mikeio.Dfsu, mikeio.Dfs2]
