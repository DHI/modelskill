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
        item: str = None,
        itemInfo=None,
        name: str = None,
        quantity: str = None,
        **kwargs,
    ) -> None:
        assert isinstance(
            data, get_args(types.UnstructuredType)
        ), "Could not construct DfsuModelResult from provided data"
        if isinstance(data, (str, Path)):
            assert Path(data).suffix == ".dfsu", "File must be a dfsu file"
            data = mikeio.open(data)
            if itemInfo is None:
                item, idx = utils.get_item_name_and_idx_dfs(data, item)
                itemInfo = data.items[idx].type

        elif isinstance(data, mikeio.DataArray):
            raise NotImplementedError  # TODO: implement this, need example
        elif isinstance(data, mikeio.Dataset):
            raise NotImplementedError  # TODO: implement this, need example

        elif isinstance(data, mikeio.spatial.FM_geometry.GeometryFM):
            raise NotImplementedError  # What to do here?

        if itemInfo is None:
            itemInfo = mikeio.EUMType.Undefined

        super().__init__(data, item, itemInfo, name, quantity)

    def extract(
        self, observation: Union[PointObservation, TrackObservation]
    ) -> protocols.Comparable:

        type_extraction_mapping = {
            (mikeio.dfsu.Dfsu2DH, PointObservation): extraction.point_obs_from_dfsu_mr,
            (mikeio.dfsu.Dfsu2DH, TrackObservation): extraction.track_obs_from_dfsu_mr,
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


if __name__ == "__main__":
    dfsu = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    test = DfsuModelResult(
        dfsu,
        item="Surface elevation",
        name="test",
        itemInfo=mikeio.EUMType.Significant_wave_height,
    )

    assert isinstance(test, protocols.ModelResult)
    assert isinstance(test, protocols.Extractable)

    dfsu_data = mikeio.open("tests/testdata/Oresund2D.dfsu")
    point_obs = PointObservation(
        "tests/testdata/SW/HKNA_Hm0.dfs0", item=0, x=4.2420, y=52.6887, name="HKNA"
    )
    track_obs = TrackObservation(
        "tests/testdata/SW/Alti_c2_Dutch.dfs0", item=3, name="C2"
    )

    test.extract(point_obs)
    test.extract(track_obs)
