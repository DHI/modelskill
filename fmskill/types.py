from enum import Enum, auto
from pathlib import Path
from typing import Optional, Union, List

import mikeio
import pandas as pd
import xarray as xr


class GeometryType(Enum):
    """Geometry type (gtype) of data"""

    POINT = auto()
    TRACK = auto()
    UNSTRUCTURED = auto()
    GRID = auto()

    def __str__(self) -> str:
        return self.name.lower()

    @staticmethod
    def from_string(s: str) -> "GeometryType":
        """Convert string to GeometryType

        Examples
        --------
        >>> GeometryType.from_string("point")
        <GeometryType.POINT: 1>
        >>> GeometryType.from_string("track")
        <GeometryType.TRACK: 2>
        >>> GeometryType.from_string("unstructured")
        <GeometryType.UNSTRUCTURED: 3>
        >>> GeometryType.from_string("flexible mesh")
        <GeometryType.UNSTRUCTURED: 3>
        >>> GeometryType.from_string("dfsu")
        <GeometryType.UNSTRUCTURED: 3>
        >>> GeometryType.from_string("grid")
        <GeometryType.GRID: 4>
        """

        try:
            if ("flex" in s.lower()) or ("dfsu" in s.lower()):
                return GeometryType.UNSTRUCTURED

            return GeometryType[s.upper()]
        except KeyError as e:
            raise KeyError(
                f"GeometryType {s} not recognized. Available options: {[m.name for m in GeometryType]}"
            ) from e


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
]
GridType = Union[str, Path, List, xr.Dataset, xr.DataArray]

PointType = Union[str, Path, pd.DataFrame, mikeio.Dfs0]

DfsType = Union[mikeio.Dfs0, mikeio.Dfsu]

ItemType = Optional[Union[str, int]]
