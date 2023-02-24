import os
from enum import Enum, auto
from pathlib import Path
from typing import Literal, Optional

import mikeio
import pandas as pd
import xarray as xr

from fmskill import model, types, utils


class ModelType(Enum):
    Point = (auto(),)
    Track = (auto(),)
    Unstructured = (auto(),)
    Grid = auto()

    def __str__(self) -> str:
        return self.name.lower()

    def from_string(s: str) -> "ModelType":
        try:
            return ModelType[s.capitalize()]
        except KeyError as e:
            raise KeyError(
                f"ModelType {s} not recognized. Available options: {[m.name for m in ModelType]}"
            ) from e


type_lookup = {
    ModelType.Point: model.PointModelResult,
    ModelType.Track: model.TrackModelResult,
    ModelType.Unstructured: model.DfsuModelResult,
    ModelType.Grid: model.GridModelResult,
}


class ModelResult:
    def __new__(
        cls,
        data: types.DataInputType,
        model_type: Optional[Literal["point", "track", "unstructured", "grid"]] = None,
        item: Optional[str] = None,
        name: Optional[str] = None,
        quantity: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ):
        if isinstance(data, (str, Path)):
            data = Path(data)
            file_ext = data.suffix.lower()
        else:
            file_ext = None

        if file_ext in [".dfsu", ".dfs0"]:
            data = mikeio.open(data)
            item = utils.get_item_name_dfs(data, item)

        elif file_ext == ".nc":
            data = xr.open_dataset(data)

        if isinstance(data, xr.Dataset) and (model_type is None):
            return model.GridModelResult(data, item, name, quantity)

        elif file_ext == ".dfsu" and (model_type is None):
            return model.DfsuModelResult(data, item, name, quantity)

        elif file_ext == ".dfs0":
            data = data.read()
            present_variables = [c.name for c in data.items]
            coord_matches = [
                c
                for c in present_variables
                if c.lower() in utils.POS_COORDINATE_NAME_MAPPING.keys()
            ]
            if coord_matches:
                data = data[coord_matches + [item]].to_dataframe().dropna()
                if model_type is None:
                    return model.TrackModelResult(
                        data=data,
                        item=item,
                        name=name,
                        quantity=quantity,
                    )
            else:
                mikeio.DataArray
                data = data[item]._to_dataset().to_dataframe().dropna()
                if model_type is None:
                    return model.PointModelResult(
                        data=data,
                        item=item,
                        name=name,
                        quantity=quantity,
                        x=x,
                        y=y,
                    )

        if model_type is not None:
            model_type = ModelType.from_string(model_type)
            return type_lookup[model_type](data, item, name, quantity)


if __name__ == "__main__":
    mr1 = ModelResult("tests/testdata/Oresund2D.dfsu", item="Surface elevation")
    assert isinstance(mr1, model.DfsuModelResult)
    mr2 = ModelResult("tests/testdata/SW/Alti_c2_Dutch.dfs0", item="swh")
    assert isinstance(mr2, model.TrackModelResult)
    mr3 = ModelResult("tests/testdata/SW/ERA5_DutchCoast.nc", item="swh")
    assert isinstance(mr3, model.GridModelResult)
    mr4 = ModelResult("tests/testdata/SW/eur_Hm0.dfs0", item="Hm0")
    assert isinstance(mr4, model.PointModelResult)
