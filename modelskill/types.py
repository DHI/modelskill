from enum import Enum, auto
from pathlib import Path
from typing import Optional, Union, List

from dataclasses import dataclass
import warnings

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
    mikeio.Dataset,
    mikeio.DataArray,
]
GridType = Union[str, Path, List, xr.Dataset, xr.DataArray]

PointType = Union[
    str, Path, pd.DataFrame, pd.Series, mikeio.Dfs0, mikeio.DataArray, mikeio.Dataset
]
TrackType = Union[str, Path, pd.DataFrame, mikeio.Dfs0, mikeio.Dataset]

# DfsType = Union[mikeio.Dfs0, mikeio.Dfsu]

ItemType = Optional[Union[str, int]]


@dataclass(frozen=True)
class Quantity:
    name: str
    unit: str

    def __str__(self):
        return f"{self.name} [{self.unit}]"

    def is_compatible(self, other) -> bool:
        """Check if the quantity is compatible with another quantity

        Examples
        --------
        >>> wl = Quantity(name="Water Level", unit="meter")
        >>> ws = Quantity(name="Wind Speed", unit="meter per second")
        >>> wl.is_compatible(ws)
        False
        >>> uq = Quantity(name="Undefined", unit="Undefined")
        >>> wl.is_compatible(uq)
        True
        """

        if self == other:
            return True

        if (self.name == "Undefined") or (other.name == "Undefined"):
            return True

        return False

    @staticmethod
    def undefined():
        return Quantity(name="Undefined", unit="Undefined")

    @staticmethod
    def from_mikeio_iteminfo(iteminfo: mikeio.ItemInfo):
        return Quantity(name=repr(iteminfo.type), unit=iteminfo.unit.name)

    @staticmethod
    def from_mikeio_eum_name(type_name: str):
        """Create a Quantity from a name recognized by mikeio

        Parameters
        ----------
        type_name : str
            Name of the quantity

        Examples
        --------
        >>> Quantity.from_mikeio_eum_name("Water Level")
        Quantity(name='Water Level', unit='meter')
        """
        try:
            etype = mikeio.EUMType[type_name]
        except KeyError:
            name_underscore = type_name.replace(" ", "_")
            try:
                etype = mikeio.EUMType[name_underscore]
            except KeyError:
                raise ValueError(
                    f"{type_name=} is not recognized as a known type. Please create a Quantity(name='{type_name}' unit='<FILL IN UNIT>')"
                )
        unit = etype.units[0].name
        warnings.warn(f"{unit=} was automatically set for {type_name=}")
        return Quantity(name=type_name, unit=unit)
