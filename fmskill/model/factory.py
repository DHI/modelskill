import os
from enum import Enum, auto
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import mikeio
import pandas as pd
import xarray as xr

from fmskill import model, types, utils
from fmskill.model.point import PointModelResult
from fmskill.model.track import TrackModelResult


class ResultGeomType(Enum):
    Point = (auto(),)
    Track = (auto(),)
    Unstructured = (auto(),)
    Grid = auto()

    def __str__(self) -> str:
        return self.name.lower()

    def from_string(s: str) -> "ResultGeomType":
        try:
            return ResultGeomType[s.capitalize()]
        except KeyError as e:
            raise KeyError(
                f"ModelType {s} not recognized. Available options: {[m.name for m in ResultGeomType]}"
            ) from e


type_lookup = {
    ResultGeomType.Point: model.PointModelResult,
    ResultGeomType.Track: model.TrackModelResult,
    ResultGeomType.Unstructured: model.DfsuModelResult,
    ResultGeomType.Grid: model.GridModelResult,
}


class ModelResult:
    def __new__(
        cls,
        data: types.DataInputType,
        geometry_type: Optional[
            Literal["point", "track", "unstructured", "grid"]
        ] = None,
        item: types.ItemType = None,
        itemInfo=None,
        name: Optional[str] = None,
        quantity: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ):
        if isinstance(data, (str, Path)):
            data = Path(data)
            file_ext = data.suffix.lower()
            if name is None:
                name = data.stem
        else:
            file_ext = None

        if geometry_type is not None:
            geometry_type = ResultGeomType.from_string(geometry_type)
            return type_lookup[geometry_type](
                data=data,
                item=item,
                itemInfo=itemInfo,
                name=name,
                quantity=quantity,
                x=x,
                y=y,
            )

        if (file_ext == ".dfsu") or isinstance(
            data,
            (mikeio.Dataset, mikeio.dfsu._Dfsu, mikeio.spatial.FM_geometry.GeometryFM),
        ):
            return model.DfsuModelResult(
                data=data, item=item, itemInfo=itemInfo, name=name, quantity=quantity
            )

        if (file_ext == ".nc") or isinstance(data, (list, xr.Dataset, xr.DataArray)):
            return model.GridModelResult(
                data=data, item=item, itemInfo=itemInfo, name=name, quantity=quantity
            )

        if (file_ext == ".dfs0") or isinstance(data, (pd.DataFrame, mikeio.Dfs0)):
            return model.PointModelResult(
                data=data,
                item=item,
                itemInfo=itemInfo,
                name=name,
                quantity=quantity,
                x=x,
                y=y,
            )


if __name__ == "__main__":
    mr1 = ModelResult("tests/testdata/Oresund2D.dfsu", item="Surface elevation")
    assert isinstance(mr1, model.DfsuModelResult)
    # mr2 = ModelResult("tests/testdata/SW/Alti_c2_Dutch.dfs0", item="swh")
    # assert isinstance(mr2, model.TrackModelResult)
    mr3 = ModelResult("tests/testdata/SW/ERA5_DutchCoast.nc", item="swh")
    assert isinstance(mr3, model.GridModelResult)
    mr4 = ModelResult("tests/testdata/SW/eur_Hm0.dfs0", item="Hm0", x=12.5, y=55.5)
    assert isinstance(mr4, model.PointModelResult)
    print("hold")
