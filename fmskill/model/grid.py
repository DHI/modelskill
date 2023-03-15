from pathlib import Path
from typing import Optional, Sequence, Union

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
        *,
        name: Optional[str] = None,
        item: Optional[Union[str, int]] = None,
        itemInfo=None,
        quantity: Optional[str] = None,
    ) -> None:
        # assert isinstance(
        #     data, types.GridType
        # ), "Could not construct GridModelResult from provided data."

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

        assert isinstance(data, xr.Dataset)

        super().__init__(
            data=data, name=name, item=item, itemInfo=itemInfo, quantity=quantity
        )

    # def _in_domain(self, x: float, y: float) -> bool:
    #    assert hasattr(self.data, "x") and hasattr(
    #        self.data, "y"
    #    ), "Data has no x and/or y coordinates."
    #    xmin = self.data.x.values.min()
    #    xmax = self.data.x.values.max()
    #    ymin = self.data.y.values.min()
    #    ymax = self.data.y.values.max()
    #    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    def extract(
        self, observation: Union[PointObservation, TrackObservation]
    ) -> protocols.Comparable:
        type_extraction_mapping = {
            (xr.Dataset, PointObservation): extraction.extract_point_from_xr,
            (xr.Dataset, TrackObservation): extraction.extract_track_from_xr,
            # (mikeio.Dfs2, PointObservation): None,  # Possible future work
            # (mikeio.Dfs2, TrackObservation): None,  # Possible future work
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
