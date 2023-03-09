from pathlib import Path
from typing import Optional, Union, List

import mikeio
import pandas as pd
import xarray as xr

DataInputType = Union[
    str,
    Path,
    List[str],
    List[Path],
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

UnstructuredType = Union[
    str,
    Path,
    mikeio.dfsu.Dfsu2DH,
    mikeio.Dataset,
    mikeio.DataArray,
    mikeio.spatial.FM_geometry.GeometryFM,
]
GridType = Union[str, Path, List, xr.Dataset, xr.DataArray]

PointType = Union[str, Path, pd.DataFrame, mikeio.Dfs0]

DfsType = Union[mikeio.Dfs0, mikeio.Dfsu]

ItemType = Optional[Union[str, int]]
