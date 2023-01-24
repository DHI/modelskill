from textwrap import dedent
import xarray as xr

from fmskill import parsing, types
from typing import Union, Optional


class DataContainer:
    """
    Class to hold data from either a model result or an observation.
    This is not a user facing class, but is used internally by fmskill.
    If the data is provided as either a .dfs or netcdf file, then the data-field of the class
    remains None.
    Any other input format will be parsed into an xr.DataArray and stored in the data-field.

    Multiple DataContainers (either containing multiple model results or a mix of model results and observations)
    can be used for comparison.
    """

    def __init__(
        self,
        data: types.DataInputType,
        item: types.ItemSpecifier = None,
        is_result: Optional[bool] = None,
        is_observation: Optional[bool] = None,
    ) -> None:

        if [is_result, is_observation].count(True) != 1:
            raise ValueError("One of is_result or is_observation must be set to True.")

        # Attribute declarations for overview, these will be filled during initialization
        self.data: Union[xr.DataArray, types.LazyLoadingType] = None
        self.is_track: bool = None
        self.is_point: bool = None
        self.is_lazy: bool = None

        self.is_result: bool = is_result or not is_observation
        self.is_observation: bool = is_observation or not is_result

        parsing.validate_input_data(data, item)

        # store item for later use in case of lazy loading
        self.item = item

        self._load_data(data, item)
        self._check_point_or_track()

    def __repr__(self) -> str:
        _res = "result" if self.is_result else "observation"
        _lazy = "lazy" if self.is_lazy else "eager"
        if self.is_track:
            _type = "track data"
        elif self.is_point:
            _type = "point data"
        else:
            _type = "unknown data type"

        return f"DataContainer({_res}, {_lazy} loading, {_type})"

    def _load_data(self, data, item):
        """
        Checks if provided input data should be loaded eagerly or lazily.
        In case of eager loading, the data is loaded into an xr.DataArray and stored in self.data.
        In case of lazy loading, the data is stored in self.data as a dfsu or dfs0 object,
        to be loaded later, once obersevation points are known.
        """

        _eager_loader = parsing.get_eager_loader(data)
        if _eager_loader is not None:
            self.is_lazy = False
            _data = _eager_loader(data, item)
            self.data = parsing.validate_and_format_xarray(_data)
        else:
            self.is_lazy = True
            _lazy_loader = parsing.get_lazy_loader(data)
            self.data = _lazy_loader(data, item)

    def _check_point_or_track(self):
        """
        Sets self.is_point and self.is_track, depending on if self.data contains multiple
        unique coordinates or not.
        """
        if self.is_lazy:
            return

        if self.data.x.size > 1 or self.data.y.size > 1:
            self.is_point, self.is_track = False, True
        else:
            self.is_point, self.is_track = True, False


if __name__ == "__main__":
    fn_1 = "tests/testdata/SW/ERA5_DutchCoast.nc"
    fn_2 = "tests/testdata/Oresund2D.dfsu"
    fn_3 = "tests/testdata/SW/Alti_c2_Dutch.dfs0"

    dc_1 = DataContainer(fn_1, item=0, is_result=True)
    dc_2 = DataContainer(fn_2, item=0, is_result=True)
    dc_3 = DataContainer(fn_3, item="swh", is_observation=True)

    print(dc_1.data)
