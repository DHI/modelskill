from enum import Enum, auto
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import xarray as xr

from fmskill import model, types  # , utils


class GeomType(Enum):
    """Geometry type of model result"""

    POINT = auto()
    TRACK = auto()
    UNSTRUCTURED = auto()
    GRID = auto()

    def __str__(self) -> str:
        return self.name.lower()

    @staticmethod
    def from_string(s: str) -> "GeomType":
        """Convert string to GeomType

        Examples
        --------
        >>> GeomType.from_string("point")
        <GeomType.POINT: 1>
        >>> GeomType.from_string("track")
        <GeomType.TRACK: 2>
        >>> GeomType.from_string("unstructured")
        <GeomType.UNSTRUCTURED: 3>
        >>> GeomType.from_string("grid")
        <GeomType.GRID: 4>
        """

        try:
            return GeomType[s.upper()]
        except KeyError as e:
            raise KeyError(
                f"GeomType {s} not recognized. Available options: {[m.name for m in GeomType]}"
            ) from e


type_lookup = {
    GeomType.POINT: model.PointModelResult,
    GeomType.TRACK: model.TrackModelResult,
    GeomType.UNSTRUCTURED: model.DfsuModelResult,
    GeomType.GRID: model.GridModelResult,
}


class ModelResult:
    def __new__(
        cls,
        data: types.DataInputType,
        *,
        geometry_type: Optional[
            Literal["point", "track", "unstructured", "grid"]
        ] = None,
        **kwargs,
    ):
        if geometry_type is None:
            geometry_type = cls._guess_geometry_type(data)
        else:
            geometry_type = GeomType.from_string(geometry_type)

        return type_lookup[geometry_type](
            data=data,
            **kwargs,
        )

        # if (file_ext == ".dfsu") or isinstance(
        #     data,
        #     (
        #         mikeio.Dataset,
        #         mikeio.DataArray,
        #         mikeio.dfsu._Dfsu,
        #         mikeio.spatial.FM_geometry.GeometryFM,
        #     ),
        # ):
        #     return model.DfsuModelResult(
        #         data=data, item=item, itemInfo=itemInfo, name=name, quantity=quantity
        #     )

        # if (file_ext == ".nc") or isinstance(data, (list, xr.Dataset, xr.DataArray)):
        #     return model.GridModelResult(
        #         data=data, item=item, itemInfo=itemInfo, name=name, quantity=quantity
        #     )

        # if (file_ext == ".dfs0") or isinstance(data, (pd.DataFrame, mikeio.Dfs0)):
        #     return model.PointModelResult(
        #         data=data,
        #         item=item,
        #         itemInfo=itemInfo,
        #         name=name,
        #         quantity=quantity,
        #         x=x,
        #         y=y,
        #     )

    @staticmethod
    def _guess_geometry_type(data) -> GeomType:

        if hasattr(data, "geometry"):
            geom_str = repr(data.geometry).lower()
            if "flex" in geom_str:
                return GeomType.UNSTRUCTURED
            elif "point" in geom_str:
                return GeomType.POINT
            else:
                raise ValueError(
                    "Could not guess geometry_type from geometry, please speficy geometry_type, e.g. geometry_type='track'"
                )

        if isinstance(data, (str, Path)):
            data = Path(data)
            file_ext = data.suffix.lower()
            if file_ext == ".dfsu":
                return GeomType.UNSTRUCTURED
            elif file_ext == ".dfs0":
                # could also be a track, but we don't know
                return GeomType.POINT
            elif file_ext == ".nc":
                # could also be point or track, but we don't know
                return GeomType.GRID
            else:
                raise ValueError(
                    "Could not guess geometry_type from file extension, please speficy geometry_type, e.g. geometry_type='track'"
                )

        if isinstance(data, (xr.Dataset, xr.DataArray)):
            if len(data.coords) >= 3:
                # if DataArray use ndim instead
                return GeomType.GRID
            else:
                raise ValueError("Could not guess geometry_type from xarray object")
        if isinstance(data, (pd.DataFrame, pd.Series)):
            # could also be a track, but we don't know
            return GeomType.POINT

        raise ValueError("Geometry type could not be guessed from this type of data")


# if __name__ == "__main__":
#     mr1 = ModelResult("tests/testdata/Oresund2D.dfsu", item="Surface elevation")
#     assert isinstance(mr1, model.DfsuModelResult)
#     # mr2 = ModelResult("tests/testdata/SW/Alti_c2_Dutch.dfs0", item="swh")
#     # assert isinstance(mr2, model.TrackModelResult)
#     mr3 = ModelResult("tests/testdata/SW/ERA5_DutchCoast.nc", item="swh")
#     assert isinstance(mr3, model.GridModelResult)
#     mr4 = ModelResult("tests/testdata/SW/eur_Hm0.dfs0", item="Hm0", x=12.5, y=55.5)
#     assert isinstance(mr4, model.PointModelResult)
#     print("hold")
