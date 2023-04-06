from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union, get_args
import xarray as xr

import mikeio

from fmskill import types, utils
from fmskill.model import protocols, PointModelResult, TrackModelResult
from fmskill.model._base import ModelResultBase
from fmskill.observation import (
    Point,
    Track,
)


class GridModelResult(ModelResultBase):
    """Construct a GridModelResult from a file or xarray.Dataset.

    Parameters
    ----------
    data : types.GridType
        the input data or file path
    name : Optional[str], optional
        The name of the model result,
        by default None (will be set to file name or item name)
    item : Optional[Union[str, int]], optional
        If multiple items/arrays are present in the input an item
        must be given (as either an index or a string), by default None
    itemInfo : Optional[mikeio.ItemInfo], optional
        Optionally, a MIKE IO ItemInfo (MIKE EUM system)
        can be given to define the type and unit of the quantity, by default None
    quantity : Optional[str], optional
        A string to identify the quantity, by default None
    """

    def __init__(
        self,
        data: types.GridType,
        *,
        name: Optional[str] = None,
        item: Optional[Union[str, int]] = None,
        itemInfo: Optional[mikeio.ItemInfo] = None,
        quantity: Optional[str] = None,
    ) -> None:
        assert isinstance(
            data, get_args(types.GridType)
        ), "Could not construct GridModelResult from provided data."

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

    def _in_domain(self, x: float, y: float) -> bool:
        assert hasattr(self.data, "x") and hasattr(
            self.data, "y"
        ), "Data has no x and/or y coordinates."
        xmin = self.data.x.values.min()
        xmax = self.data.x.values.max()
        ymin = self.data.y.values.min()
        ymax = self.data.y.values.max()
        return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    def extract(self, pos: Union[Point, Track]) -> protocols.Comparable:
        """Extract ModelResult at observation positions

        Parameters
        ----------
        observation : <PointObservation> or <TrackObservation>
            positions (and times) at which modelresult should be extracted

        Returns
        -------
        <fmskill.protocols.Comparable>
            A model result object with the same geometry as the observation
        """
        extractor_lookup = {
            Point: self._extract_point,
            Track: self._extract_track,
        }
        extraction_func = extractor_lookup.get(type(pos))
        if extraction_func is None:
            raise NotImplementedError(
                f"Extraction from {type(self.data)} to {type(pos)} is not implemented."
            )
        return extraction_func(pos)

    def _extract_point(self, point: Point) -> PointModelResult:
        """Spatially extract a PointModelResult from a GridModelResult (when data is a xarray.Dataset),
        given a Point. No time interpolation is done!"""

        x, y = point.x, point.y
        if (x is None) or (y is None):
            raise ValueError(
                f"PointObservation '{point.name}' cannot be used for extraction "
                + f"because it has None position x={x}, y={y}. Please provide position "
                + "when creating PointObservation."
            )
        if not self._in_domain(x, y):
            raise ValueError(
                f"PointObservation '{point.name}' ({x}, {y}) is outside model domain!"
            )

        # self._validate_any_obs_in_model_time(
        #    point.name, point.data.index, self.time
        # )

        da = self.data[self.item].interp(coords=dict(x=x, y=y), method="nearest")
        df = da.to_dataframe().drop(columns=["x", "y"])
        df = df.rename(columns={df.columns[-1]: self.name})

        return PointModelResult(
            data=df.dropna(),
            x=da.x.item(),
            y=da.y.item(),
            item=self.name,
            itemInfo=self.itemInfo,
            name=self.name,
            quantity=self.quantity,
        )

    def _extract_track(self, track: Track) -> TrackModelResult:
        """Extract a TrackModelResult from a GridModelResult (when data is a xarray.Dataset),
        given a TrackObservation."""

        # self._validate_any_obs_in_model_time(track.name, track.data.index, self.time)

        renamed_obs_data = utils.rename_coords_pd(track.data)
        t = xr.DataArray(renamed_obs_data.index, dims="track")
        x = xr.DataArray(renamed_obs_data.x, dims="track")
        y = xr.DataArray(renamed_obs_data.y, dims="track")
        da = self.data[self.item].interp(coords=dict(time=t, x=x, y=y), method="linear")
        df = da.to_dataframe().drop(columns=["time"])
        # df.index.name = "time"
        df = df.rename(columns={df.columns[-1]: self.name})

        return TrackModelResult(
            data=df.dropna(),
            item=self.name,
            itemInfo=self.itemInfo,
            name=self.name,
            quantity=self.quantity,
        )
