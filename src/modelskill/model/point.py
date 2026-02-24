from __future__ import annotations
from typing import Sequence, Any

import xarray as xr

from ..obs import Observation
from ..types import PointType
from ..quantity import Quantity
from ..timeseries import TimeSeries, _parse_xyz_point_input
from ..timeseries._align import align_data


class PointModelResult(TimeSeries):
    """Model result for a single point location.

    Construct a PointModelResult from a 0d data source:
    dfs0 file, mikeio.Dataset/DataArray, pandas.DataFrame/Series
    or xarray.Dataset/DataArray

    Parameters
    ----------
    data : str, Path, mikeio.Dataset, mikeio.DataArray, pd.DataFrame, pd.Series, xr.Dataset or xr.DataArray
        filename (.dfs0 or .nc) or object with the data
    name : Optional[str], optional
        The name of the model result,
        by default None (will be set to file name or item name)
    x : float, optional
        first coordinate of point position, inferred from data if not given, else None
    y : float, optional
        second coordinate of point position, inferred from data if not given, else None
    z : float, optional
        third coordinate of point position, inferred from data if not given, else None
    item : str | int | None, optional
        If multiple items/arrays are present in the input an item
        must be given (as either an index or a string), by default None
    quantity : Quantity, optional
        Model quantity, for MIKE files this is inferred from the EUM information
    aux_items : Optional[list[int | str]], optional
        Auxiliary items, by default None
    """

    def __init__(
        self,
        data: PointType,
        *,
        name: str | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        item: str | int | None = None,
        quantity: Quantity | None = None,
        aux_items: Sequence[int | str] | None = None,
    ) -> None:
        if not self._is_input_validated(data):
            data = _parse_xyz_point_input(
                data,
                name=name,
                item=item,
                quantity=quantity,
                aux_items=aux_items,
                x=x,
                y=y,
                z=z,
            )

        assert isinstance(data, xr.Dataset)

        data_var = str(list(data.data_vars)[0])
        data[data_var].attrs["kind"] = "model"
        super().__init__(data=data)

    def interp_time(self, observation: Observation, **kwargs: Any) -> PointModelResult:
        """
        Interpolate model result to the time of the observation

        wrapper around xarray.Dataset.interp()

        Parameters
        ----------
        observation : Observation
            The observation to interpolate to
        **kwargs

            Additional keyword arguments passed to xarray.interp

        Returns
        -------
        PointModelResult
            Interpolated model result
        """
        ds = align_data(self.data, observation, **kwargs)
        return PointModelResult(ds)
