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
        model_type: Optional[Literal["point", "track", "unstructured", "grid"]] = None,
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

        if file_ext in [".dfsu", ".dfs0"]:
            data = mikeio.open(data)
            item, idx = utils.get_item_name_and_idx_dfs(data, item)
            itemInfo = data.items[idx].type

        elif (file_ext == ".nc") or isinstance(data, (list, xr.Dataset)):
            data, item, itemInfo = cls._xarray_handler(data, item)

        if isinstance(data, xr.Dataset) and (model_type is None):
            return model.GridModelResult(
                data=data, item=item, itemInfo=itemInfo, name=name, quantity=quantity
            )

        elif file_ext == ".dfsu" and (model_type is None):
            return model.DfsuModelResult(
                data=data, item=item, itemInfo=itemInfo, name=name, quantity=quantity
            )

        elif file_ext == ".dfs0":
            data, mr_type = cls._dfs0_handler(data, item)

        elif isinstance(data, pd.DataFrame):
            data, mr_type = cls._pandas_handler(data, item)
        else:
            mr_type = None

        if (mr_type == ResultGeomType.Track) and (model_type is None):
            return model.TrackModelResult(
                data=data,
                item=item,
                itemInfo=itemInfo,
                name=name,
                quantity=quantity,
            )
        elif (mr_type == ResultGeomType.Point) and (model_type is None):
            return model.PointModelResult(
                data=data,
                item=item,
                itemInfo=itemInfo,
                name=name,
                quantity=quantity,
                x=x,
                y=y,
            )

        if model_type is not None:
            model_type = ResultGeomType.from_string(model_type)
            return type_lookup[model_type](data, item, name, quantity)

    @staticmethod
    def _xarray_handler(
        data: Union[str, Path, xr.Dataset, List[str], List[Path]], item: types.ItemType
    ) -> Tuple[xr.Dataset, str, mikeio.EUMType]:
        if isinstance(data, (str, Path)):
            data = xr.open_dataset(data)
        elif isinstance(data, list):
            if all(Path(f).suffix == ".nc" for f in data if isinstance(f, (str, Path))):
                data = xr.open_mfdataset(data)
            else:
                raise ValueError("List of files must be netCDF files")

        item, idx = utils.get_item_name_and_idx_xr(data, item)
        itemInfo = mikeio.EUMType.Undefined

        return data, item, itemInfo

    @staticmethod
    def _dfs0_handler(
        data: mikeio.Dfs0, item: types.ItemType
    ) -> Tuple[pd.DataFrame, ResultGeomType]:
        """
        Checks if the Dfs0 object contains a track or point result, selects the relevant
        items and returns a pandas DataFrame.
        """
        data = data.read()
        present_variables = [c.name for c in data.items]
        coord_matches = [
            c
            for c in present_variables
            if c.lower() in utils.POS_COORDINATE_NAME_MAPPING.keys()
        ]
        if coord_matches:
            data = data[coord_matches + [item]].to_dataframe().dropna()
            return data, ResultGeomType.Track
        else:
            data = data[item]._to_dataset().to_dataframe().dropna()
            return data, ResultGeomType.Point

    @staticmethod
    def _pandas_handler(
        data: pd.DataFrame, item: types.ItemType
    ) -> Tuple[pd.DataFrame, ResultGeomType]:
        """
        Checks if the DataFrame object contains a track or point result, selects the relevant
        items and returns a pandas DataFrame.
        """
        present_variables = list(data.columns)
        coord_matches = [
            c
            for c in present_variables
            if c.lower() in utils.POS_COORDINATE_NAME_MAPPING.keys()
        ]
        if coord_matches:
            data = data[coord_matches + [item]].dropna()
            return data, ResultGeomType.Track
        else:
            data = data[[item]].dropna()
            return data, ResultGeomType.Point


if __name__ == "__main__":
    mr1 = ModelResult("tests/testdata/Oresund2D.dfsu", item="Surface elevation")
    assert isinstance(mr1, model.DfsuModelResult)
    mr2 = ModelResult("tests/testdata/SW/Alti_c2_Dutch.dfs0", item="swh")
    assert isinstance(mr2, model.TrackModelResult)
    mr3 = ModelResult("tests/testdata/SW/ERA5_DutchCoast.nc", item="swh")
    assert isinstance(mr3, model.GridModelResult)
    mr4 = ModelResult("tests/testdata/SW/eur_Hm0.dfs0", item="Hm0", x=12.5, y=55.5)
    assert isinstance(mr4, model.PointModelResult)
