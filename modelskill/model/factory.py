from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional, Any
import warnings

import pandas as pd
import xarray as xr

from .point import PointModelResult
from .track import TrackModelResult
from .dfsu import DfsuModelResult
from .grid import GridModelResult


from ..types import GeometryType, DataInputType

_modelresult_lookup = {
    GeometryType.POINT: PointModelResult,
    GeometryType.TRACK: TrackModelResult,
    GeometryType.UNSTRUCTURED: DfsuModelResult,
    GeometryType.GRID: GridModelResult,
}


def model_result(
    data: DataInputType,
    *,
    aux_items: Optional[list[int | str]] = None,
    gtype: Optional[Literal["point", "track", "unstructured", "grid"]] = None,
    **kwargs: Any,
) -> Any:
    """A factory function for creating an appropriate object based on the data input.

    Parameters
    ----------
    data : DataInputType
        The data to be used for creating the ModelResult object.
    aux_items : Optional[list[int | str]]
        Auxiliary items, by default None
    gtype : Optional[Literal["point", "track", "unstructured", "grid"]]
        The geometry type of the data. If not specified, it will be guessed from the data.
    **kwargs
        Additional keyword arguments to be passed to the ModelResult constructor.

    Examples
    --------
    >>> import modelskill as ms
    >>> ms.model_result("Oresund2D.dfsu", item=0)
    <DfsuModelResult> 'Oresund2D'
    >>> ms.model_result("ERA5_DutchCoast.nc", item="swh", name="ERA5")
    <GridModelResult> 'ERA5'
    """
    if gtype is None:
        geometry = _guess_gtype(data)
    else:
        geometry = GeometryType.from_string(gtype)

    return _modelresult_lookup[geometry](
        data=data,
        aux_items=aux_items,
        **kwargs,
    )


class ModelResult:
    def __new__(
        cls,
        data: DataInputType,
        *,
        gtype: Optional[Literal["point", "track", "unstructured", "grid"]] = None,
        **kwargs: Any,
    ) -> Any:
        # deprecated
        warnings.warn(
            FutureWarning(
                "Use model_result or an explicit class instead, e.g. PointModelResult"
            )
        )
        if gtype is None:
            geometry = _guess_gtype(data)
        else:
            geometry = GeometryType.from_string(gtype)

        return _modelresult_lookup[geometry](
            data=data,
            **kwargs,
        )


def _guess_gtype(data: Any) -> GeometryType:
    if hasattr(data, "geometry"):
        geom_str = repr(data.geometry).lower()
        if "flex" in geom_str:
            return GeometryType.UNSTRUCTURED
        elif "point" in geom_str:
            return GeometryType.POINT
        else:
            raise ValueError(
                "Could not guess gtype from geometry, please specify gtype, e.g. gtype='track'"
            )

    if isinstance(data, (str, Path)):
        data = Path(data)
        file_ext = data.suffix.lower()
        if file_ext == ".dfsu":
            return GeometryType.UNSTRUCTURED
        elif file_ext == ".dfs0":
            # could also be a track, but we don't know
            return GeometryType.POINT
        elif file_ext == ".nc":
            # could also be point or track, but we don't know
            return GeometryType.GRID
        else:
            raise ValueError(
                "Could not guess gtype from file extension, please specify gtype, e.g. gtype='track'"
            )

    if isinstance(data, (xr.Dataset, xr.DataArray)):
        if len(data.coords) >= 3:
            # if DataArray use ndim instead
            return GeometryType.GRID
        else:
            raise ValueError("Could not guess gtype from xarray object")
    if isinstance(data, (pd.DataFrame, pd.Series)):
        # could also be a track, but we don't know
        return GeometryType.POINT

    raise ValueError(
        f"Geometry type (gtype) could not be guessed from this type of data: {type(data)}. Please specify gtype, e.g. gtype='track'"
    )
