from pathlib import Path
from typing import Sequence, Union

import mikeio
import xarray as xr

from fmskill import types, utils
from fmskill.model import extraction, protocols
from fmskill.model._base import ModelResultBase
from fmskill.observation import PointObservation, TrackObservation


class GridModelResult(ModelResultBase):
    def __init__(
        self,
        data: types.GridType,
        name: str = None,
        item: Union[str, int] = None,
        itemInfo=None,
        quantity: str = None,
        **kwargs,
    ) -> None:
        # assert isinstance(
        #     data, types.GridType
        # ), "Could not construct GridModelResult from provided data."

        # def _validate_file(file):
        #     # TODO: not needed...
        #     assert isinstance(
        #         file, (str, Path)
        #     ), f"Files need to be specified as str or Path objects."
        #     # assert Path(file).suffix == ".nc", f"{file}: Not a netcdf file."
        #     assert Path(file).exists(), f"{file}: File does not exist."

        # TODO: Dataset is missing

        # TODO: if dataset -> do nothing
        #       if dataarray -> convert to dataset
        #       if file -> open dataset
        #       if list/sequence or contains * -> open mfdataset

        name = name or super()._default_name(data)

        if isinstance(data, (str, Path)):
            if "*" in str(data):
                data = xr.open_mfdataset(data)
            else:
                assert Path(data).exists(), f"{data}: File does not exist."
                data = xr.open_dataset(data)

        elif isinstance(data, Sequence) and all(
            isinstance(file, (str, Path)) for file in data
        ):
            data = xr.open_mfdataset(data)

        elif isinstance(data, xr.DataArray):
            assert data.ndim >= 2, "DataArray must at least 2D."
            data = data.to_dataset(name=name, promote_attrs=True)
        elif isinstance(data, xr.Dataset):
            assert len(data.coords) >= 2, "Dataset must have at least 2 dimensions."

        else:
            raise NotImplementedError(
                f"Could not construct GridModelResult from {type(data)}"
            )

        item, _ = utils.get_item_name_and_idx(list(data.data_vars), item)
        name = name or item
        data = utils.rename_coords_xr(data)

        super().__init__(
            data=data, name=name, item=item, itemInfo=itemInfo, quantity=quantity
        )

    def extract(
        self, observation: Union[PointObservation, TrackObservation]
    ) -> protocols.Comparable:
        type_extraction_mapping = {
            (xr.Dataset, PointObservation): extraction.point_obs_from_xr_mr,
            (xr.Dataset, TrackObservation): extraction.extract_track_from_xr,
            (mikeio.Dfs2, PointObservation): None,  # Possible future work
            (mikeio.Dfs2, TrackObservation): None,  # Possible future work
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


# if __name__ == "__main__":
#     grid_data = "tests/testdata/SW/ERA5_DutchCoast.nc"
#     point_obs = PointObservation(
#         "tests/testdata/SW/eur_Hm0.dfs0", item=0, x=3.2760, y=51.9990, name="EPL"
#     )
#     track_obs = TrackObservation(
#         "tests/testdata/SW/Alti_c2_Dutch.dfs0", item=3, name="c2"
#     )
#     test = GridModelResult(grid_data, item="swh", name="test")

#     assert isinstance(test, protocols.ModelResult)
#     assert isinstance(test, protocols.Extractable)

#     test.extract(point_obs)
#     test.extract(track_obs)

#     c1 = test.extract_observation(point_obs, validate=False)
#     c2 = test.extract_observation(track_obs, validate=False)
