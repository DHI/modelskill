import logging

import xarray as xr

from fmskill import parsing, types
from typing import Union, Optional

logging.basicConfig(level=logging.INFO)


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
        self.item = None
        self.is_track: bool = None
        self.is_point: bool = None
        self.is_lazy: bool = None

        self.is_result: bool = is_result or not is_observation
        self.is_observation: bool = is_observation or not is_result

        parsing.validate_input_data(data, item)

        self._load_data(data, item)
        # self._check_point_or_track()

    def __repr__(self) -> str:
        _res = "result" if self.is_result else "observation"
        _lazy = "lazy" if self.is_lazy else "eager"
        if self.is_track:
            _type = "track data"
        elif self.is_point:
            _type = "point data"
        else:
            _type = "unknown data type"

        return f"DataContainer({_res}, item: '{self.item}', {_lazy} loading, {_type})"

    def _load_data(self, data, item):
        """
        Load data and store in self.data as a standardized xr.DataArray in case
        of eager loading (all observations & eager file formats),
        or as a lazy loading object in case of lazy loading (model results stored e.g. in dfs files).
        To do so, the data is first loaded as a xr.Dataset, and then parsed into a xr.DataArray
        (as the dataset may contain data variables to be used as coordinates).
        The assumption is made that observations can always be loaded eagerly.
        """

        # eager loading
        _eager_loader = parsing.get_eager_loader(data)
        if _eager_loader is not None:
            self.is_lazy = False
            ds = _eager_loader(data)
        # lazy loading
        else:
            self.is_lazy = True
            _lazy_loader = parsing.get_lazy_loader(data)
            self.data = _lazy_loader(data)

        # special case of eager loading for observations
        if self.is_observation and self.is_lazy:
            if isinstance(self.data, types.DfsType):
                ds = self.data.read().to_xarray()
                _loader = parsing.get_eager_loader(ds)
                ds = _loader(ds)
                self.is_lazy = False

        if not self.is_lazy:
            # put all positional & temporal data into standard coordinate names
            ds = parsing.parse_ds_coords(ds)
            self._check_point_or_track(ds)

            # parse into DataArray, based on item
            self.item = parsing.xarray_get_item_name(ds, item)
            self.data = ds[self.item]

        print("hold")

    def _check_point_or_track(self, ds: xr.Dataset) -> None:
        if self.is_lazy:
            return

        # if only one coordinate is present, we have point data
        if len(ds.coords) == 1:
            self.is_point, self.is_track = True, False
            return

        # The coordinates might be present, but only have one value combination, point data
        if ds.x.size == 1 and ds.y.size == 1:
            self.is_point, self.is_track = True, False
        else:
            self.is_point, self.is_track = False, True

    @staticmethod
    def _check_compatibility(containers: list["DataContainer"]) -> None:
        """
        Checks if the provided DataContainers are compatible for comparison.
        Implemented as a static method, so it may also be used for more complex validation
        of multiple DataContainers in any higher level collection of models and observations.
        """
        if len(containers) < 2:
            return

        model_results = [c for c in containers if c.is_result]
        observations = [c for c in containers if c.is_observation]

        if not model_results:
            raise ValueError(
                "Only observations provided, please provide at least one model result."
            )

        if not observations:
            raise NotImplementedError(
                """Currently, only comparisons between model results and observations are supported.
                Only model results were provided."""
            )

        ok, not_ok = [], []
        for m in model_results:
            for o in observations:
                if o.is_track and not m.is_track:
                    not_ok.append((str(m), str(o)))
                else:
                    ok.append((str(m), str(o)))

        if not ok:
            raise ValueError(
                "No compatible model results and observations found for comparison."
            )

        if not_ok:
            for m, o in not_ok:
                logging.warning(
                    f"Can't compare track observation to point model results: {m} and {o}"
                )

    def compare(self, other: "DataContainer"):
        self._check_compatibility([self, other])


if __name__ == "__main__":
    fn_1 = "tests/testdata/SW/ERA5_DutchCoast.nc"
    fn_2 = "tests/testdata/Oresund2D.dfsu"
    fn_3 = "tests/testdata/SW/Alti_c2_Dutch.dfs0"  # track observation
    fn_4 = "tests/testdata/smhi_2095_klagshamn.dfs0"  # point observation

    dc_1 = DataContainer(fn_1, item=0, is_result=True)
    dc_2 = DataContainer(fn_2, item=0, is_result=True)
    dc_3 = DataContainer(fn_3, item="swh", is_observation=True)
    dc_4 = DataContainer(fn_4, is_observation=True)

    dc_1.compare(dc_3)

    print(dc_1.data)
