import os
from enum import Enum, auto
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import mikeio
import pandas as pd
import xarray as xr

from fmskill import model, types  # , utils

# from fmskill.model.point import PointModelResult
# from fmskill.model.track import TrackModelResult


class GeomType(Enum):
    Point = (auto(),)  # why not auto()
    Track = (auto(),)
    Unstructured = (auto(),)
    Grid = auto()

    def __str__(self) -> str:
        return self.name.lower()

    def from_string(s: str) -> "GeomType":
        try:
            return GeomType[s.capitalize()]
        except KeyError as e:
            raise KeyError(
                f"GeomType {s} not recognized. Available options: {[m.name for m in GeomType]}"
            ) from e


type_lookup = {
    GeomType.Point: model.PointModelResult,
    GeomType.Track: model.TrackModelResult,
    GeomType.Unstructured: model.DfsuModelResult,
    GeomType.Grid: model.GridModelResult,
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
            geometry_type = str(cls._guess_geometry_type(data))

        # if isinstance(data, (str, Path)):
        #     data = Path(data)
        #     file_ext = data.suffix.lower()
        #     if name is None:
        #         name = data.stem
        # else:
        #     file_ext = None

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
        fail_txt = "Could not guess geometry_type. Please provide as argument."
        if isinstance(data, (mikeio.DataArray, mikeio.Dataset, mikeio.dfsu._Dfsu)):
            if isinstance(data.geometry, mikeio.spatial.FM_geometry.GeometryFM):
                return GeomType.Unstructured
            elif isinstance(data.geometry, mikeio.spatial.Geometry.GeometryPoint2D):
                return GeomType.Point
            else:
                raise ValueError(fail_txt)

        if isinstance(data, (str, Path)):
            data = Path(data)
            file_ext = data.suffix.lower()
            if file_ext == ".dfsu":
                return GeomType.Unstructured
            elif file_ext == ".dfs0":
                # could also be a track, but we don't know
                return GeomType.Point
            elif file_ext == ".nc":
                # could also be point or track, but we don't know
                return GeomType.Grid
            else:
                raise ValueError(fail_txt)

        if isinstance(data, (xr.Dataset, xr.DataArray)):
            if len(data.coords) >= 3:
                # if DataArray use ndim instead
                return GeomType.Grid
            else:
                raise ValueError(fail_txt)
        if isinstance(data, (pd.DataFrame, pd.Series)):
            # could also be a track, but we don't know
            return GeomType.Point

        raise ValueError(fail_txt)


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
