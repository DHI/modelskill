from enum import Enum
from pathlib import Path
from typing import Union, List, Optional
from dataclasses import dataclass
import pandas as pd
import xarray as xr
import mikeio


class GeometryType(Enum):
    """Geometry type (gtype) of data"""

    POINT = "point"
    TRACK = "track"
    UNSTRUCTURED = "unstructured"
    GRID = "grid"

    def __str__(self) -> str:
        return self.name.lower()

    @staticmethod
    def from_string(s: str) -> "GeometryType":
        """Convert string to GeometryType

        Examples
        --------
        >>> from modelskill.types import GeometryType
        >>> GeometryType.from_string("point")
        <GeometryType.POINT: 'point'>
        >>> GeometryType.from_string("track")
        <GeometryType.TRACK: 'track'>
        >>> GeometryType.from_string("unstructured")
        <GeometryType.UNSTRUCTURED: 'unstructured'>
        >>> GeometryType.from_string("flexible mesh")
        <GeometryType.UNSTRUCTURED: 'unstructured'>
        >>> GeometryType.from_string("dfsu")
        <GeometryType.UNSTRUCTURED: 'unstructured'>
        >>> GeometryType.from_string("grid")
        <GeometryType.GRID: 'grid'>
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
    mikeio.Dfs0,
    mikeio.dfsu.Dfsu2DH,
    pd.DataFrame,
    pd.Series,
    xr.Dataset,
    xr.DataArray,
]

UnstructuredType = Union[
    str,
    Path,
    mikeio.dfsu.Dfsu2DH,
    mikeio.dfsu.Dfsu3D,
    mikeio.Dataset,
    mikeio.DataArray,
]
GridType = Union[str, Path, List, xr.Dataset, xr.DataArray]

PointType = Union[
    str,
    Path,
    pd.DataFrame,
    pd.Series,
    mikeio.Dfs0,
    mikeio.Dataset,
    mikeio.DataArray,
    xr.Dataset,
    xr.DataArray,
]
TrackType = Union[str, Path, pd.DataFrame, mikeio.Dfs0, mikeio.Dataset, xr.Dataset]


@dataclass(frozen=True)
class Period:
    """Period of data, defined by start and end time, can be open ended"""

    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
