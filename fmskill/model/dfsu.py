from pathlib import Path
from typing import Union, get_args

import mikeio

from fmskill import types, utils
from fmskill.model import extraction, protocols
from fmskill.model._base import ModelResultBase
from fmskill.observation import PointObservation, TrackObservation


class DfsuModelResult(ModelResultBase):
    def __init__(
        self,
        data: types.UnstructuredType,
        *,
        name: str = None,
        item: Union[str, int] = None,
        itemInfo=None,
        quantity: str = None,
    ) -> None:
        name = name or super()._default_name(data)

        assert isinstance(
            data, get_args(types.UnstructuredType)
        ), "Could not construct DfsuModelResult from provided data"

        filename = None
        if isinstance(data, (str, Path)):
            assert Path(data).suffix == ".dfsu", "File must be a dfsu file"
            filename = str(data)
            data = mikeio.open(data)

        elif isinstance(data, (mikeio.DataArray, mikeio.Dataset)):
            if not isinstance(data.geometry, mikeio.spatial.FM_geometry.GeometryFM):
                raise ValueError(f"Geometry of {type(data)} is not supported.")
        else:
            raise ValueError(
                f"data type must be .dfsu or dfsu-Dataset/DataArray. Not {type(data)}."
            )

        if isinstance(data, mikeio.DataArray):
            item = item or data.name
            itemInfo = itemInfo or data.item
        else:
            item_names = [i.name for i in data.items]
            item, idx = utils.get_item_name_and_idx(item_names, item)
            itemInfo = itemInfo or data.items[idx]

        name = name or item

        super().__init__(
            data=data, name=name, item=item, itemInfo=itemInfo, quantity=quantity
        )

        self.filename = filename  # TODO: remove? backward compatibility

    def _in_domain(self, x, y) -> bool:
        return self.data.geometry.contains([x, y])

    def extract(
        self, observation: Union[PointObservation, TrackObservation]
    ) -> protocols.Comparable:
        type_extraction_mapping = {
            (mikeio.dfsu.Dfsu2DH, PointObservation): extraction.extract_point_from_dfsu,
            (mikeio.dfsu.Dfsu2DH, TrackObservation): extraction.extract_track_from_dfsu,
            (mikeio.Dataset, PointObservation): extraction.extract_point_from_dfsu,
            (mikeio.Dataset, TrackObservation): extraction.extract_track_from_dfsu,
            (mikeio.DataArray, PointObservation): extraction.extract_point_from_dfsu,
            (mikeio.DataArray, TrackObservation): extraction.extract_track_from_dfsu,
        }

        extraction_func = type_extraction_mapping.get(
            (type(self.data), type(observation))
        )
        if extraction_func is None:
            raise NotImplementedError(
                f"Extraction from {type(self.data)} to {type(observation)} is not implemented."
            )
        extraction_result = extraction_func(self, observation)

        return extraction_result
