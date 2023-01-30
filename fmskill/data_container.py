import logging
from typing import Optional, Union

import xarray as xr

from fmskill import parsing, types

logging.basicConfig(level=logging.INFO)


class DataContainer:
    """
    Class to hold data from either a model result or an observation.
    This is not a user facing class, but is used internally by fmskill.

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
        self.data: Union[xr.Dataset, types.LazyLoadingType] = None
        self.is_field: bool = None
        self.is_track: bool = None
        self.is_point: bool = None
        self.is_dfs: bool = None

        self.is_result: bool = is_result or not is_observation
        self.is_observation: bool = is_observation or not is_result

        parsing.validate_input_data(data, item)

        self._load_data(data, item)
        if not self.is_dfs:
            self._check_field()
            self._check_point_or_track()

    @property
    def values(self):
        return self.data[self.requested_key].data

    @property
    def is_point_observation(self):
        return self.is_observation and self.is_point

    @property
    def is_track_observation(self):
        return self.is_observation and self.is_track

    def __repr__(self) -> str:
        _res = "result" if self.is_result else "observation"
        if self.is_track:
            _type = "track data"
        elif self.is_point:
            _type = "point data"
        else:
            _type = "unknown data type"

        return f"DataContainer({_res}, item: '{self.item}' {_type})"

    def _load_data(self, data, item):
        """
        Load data and store in self.data as a standardized xr.DataArray in case
        of eager loading (all observations & eager file formats),
        or as a lazy loading object in case of lazy loading (model results stored e.g. in dfs files).
        To do so, the data is first loaded as a xr.Dataset, and then parsed into a xr.DataArray
        (as the dataset may contain data variables to be used as coordinates).
        The assumption is made that observations can always be loaded eagerly.
        """

        # try loading straight into a dataset
        _eager_loader = parsing.get_dataset_loader(data)
        if _eager_loader is not None:
            self.is_dfs = False
            ds = _eager_loader(data)
        # lazily load dfs files
        else:
            self.is_dfs = True
            _lazy_loader = parsing.get_dfs_loader(data)
            self.data = _lazy_loader(data)

        # special case of observations stored in dfs files
        if self.is_observation and self.is_dfs:
            if isinstance(self.data, types.DfsType):
                ds = self.data.read().to_xarray()
                _loader = parsing.get_dataset_loader(ds)
                ds = _loader(ds)
                self.is_dfs = False

        if not self.is_dfs:
            ds = parsing.rename_coords(ds)

            self.requested_key = parsing.get_item_name(ds, item)
            self.additional_keys = parsing.get_coords_in_data_vars(ds)
            self.data = ds[self.additional_keys + [self.requested_key]]

    def _check_field(self):
        """Maybe come up with somthing better here?"""
        target_coords = ("time", "x", "y")
        present_coords = [c for c in self.data.coords if c in target_coords]
        if len(present_coords) > 1 and self.data[
            self.requested_key
        ].size == parsing._get_expected_size_if_grid(self.data[self.requested_key]):
            self.is_field = True
        else:
            self.is_field = False

    def _check_point_or_track(self) -> None:
        if self.is_dfs or self.is_field:
            return

        # combine spatial variables present in data variables and coordinates
        spatial_variables = [
            self.data.coords[c] for c in self.data.coords if c in ("x", "y")
        ] + [self.data[c] for c in self.additional_keys if c in ("x", "y")]

        if not spatial_variables:
            self.is_point, self.is_track = True, False
            return

        # The coordinates might be present, but only have one value combination, point data
        if all(d.size == 1 for d in spatial_variables):
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
        # self._check_compatibility([self, other])
        pass


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
