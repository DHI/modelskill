import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from fmskill import parsing, types

logging.basicConfig(level=logging.INFO)


class DataContainer:
    """
    Class to hold data from either a model result or an observation.
    This is not a user facing class, but is used internally by fmskill.

    Parameters
    ----------
    data : DataInputType
        Data to be stored in the DataContainer. Can be a file path, a pandas DataFrame,
        a xarray Dataset, a xarray DataArray, a mikeio DataArray or a list of file paths.
    item : ItemSpecifier, optional
        Name or index of the item to be stored in the DataContainer. Can only be None for single
        variable data
    is_result : bool, optional
        Set to True if the DataContainer is a model result, either this or is_observation must be set to True
    is_observation : bool, optional
        Set to True if the DataContainer is an observation, either this or is_result must be set to True
    x : float, optional
        x-coordinate of the observation point, should be used for point observations that do not include
        the coordinates in the data
    y : float, optional
        y-coordinate of the observation point, should be used for point observations that do not include
        the coordinates in the data
    name : str, optional
        Name of the DataContainer, will be used for plotting and logging

    Raises
    ------
    ValueError
        If neither is_result or is_observation is set to True.

    """

    def __init__(
        self,
        data: types.DataInputType,
        is_result: Optional[bool],
        item: types.ItemSpecifier = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        quantity: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:

        self.x_point = x
        self.y_point = y
        self.quantity = quantity
        self.name = name

        # Attribute declarations for overview, these will be filled during initialization
        self.data: Union[xr.Dataset, types.DfsType] = None
        self.is_field: bool = None
        self.is_track: bool = None
        self.is_point: bool = None
        self.is_dfs: bool = None
        self.item_key = None
        self.additional_keys = None

        self.is_result: bool = is_result

        parsing.validate_input_data(data, item)

        self._load_data(data, item)
        if not self.is_dfs:
            self._check_field()
            self._check_point_or_track()

    @property
    def is_observation(self):
        return not self.is_result

    @property
    def values(self):
        if not self.is_dfs:
            return self.data[self.item_key].values

    @property
    def is_point_observation(self):
        return self.is_observation and self.is_point

    @property
    def is_track_observation(self):
        return self.is_observation and self.is_track

    @property
    def x(self) -> Union[int, float, xr.DataArray, None]:
        if self.x_point is not None:
            return self.x_point
        elif not self.is_dfs:
            return self.data.x

    @property
    def y(self) -> Union[int, float, xr.DataArray, None]:
        if self.y_point is not None:
            return self.y_point
        elif not self.is_dfs:
            return self.data.y

    @property
    def time(self):
        return self.data.time

    @property
    def geometry(self):
        if self.file_extension == ".dfsu":
            return self.data.geometry

    @property
    def n_points(self):
        if self.is_point_observation:
            return len(self.time)
        elif self.is_track_observation:
            return self.values.shape[0]

    @property
    def start_time(self) -> pd.Timestamp:
        if not self.is_dfs:
            return pd.Timestamp(self.data.time[0].values)
        else:
            return self.data.start_time

    @property
    def end_time(self) -> pd.Timestamp:
        if not self.is_dfs:
            return pd.Timestamp(self.data.time[-1].values)
        else:
            return self.data.end_time

    def __repr__(self) -> str:
        def _f(_inp: str):
            return f"|{_inp.center(12)}|"

        _type = "Result" if self.is_result else "Observation"
        if not self.is_dfs:
            _unit = self.data[self.item_key].attrs.get("unit")
        else:
            _unit = None
        _unit = f" [{_unit}]" if _unit else "Unit unknown"
        if self.is_point_observation:
            _geo_type = "Point Data"
        elif self.is_track_observation:
            _geo_type = "Track Data"
        elif self.is_field:
            _geo_type = "Grid Data"
        else:
            _geo_type = ""

        return (
            f"({_f(_type)}{_f(self.name)}{_f(self.item_key)}{_f(_unit)}{_f(_geo_type)}"
        )

    def _load_data(self, data, item):
        if isinstance(data, (str, Path)):
            self.file_extension = Path(data).suffix
            if self.name is None:
                self.name = Path(data).stem
        else:
            self.file_extension = None

        # try loading straight into a dataset
        _ds_loader = parsing.get_dataset_loader(data)
        if _ds_loader is not None:
            self.is_dfs = False
            ds = _ds_loader(data)
            self.item_key, self.item_idx = parsing.get_item_name_xr_ds(ds, item)

        # load as dfs object
        else:
            self.is_dfs = True
            _dfs_loader = parsing.get_dfs_loader(data)
            self.data = _dfs_loader(data)
            self.item_key, self.item_idx = parsing.get_item_name_dfs(self.data, item)

        self.additional_keys = parsing.get_coords_in_data_vars(self.data)

        # special case of observations stored in dfs files
        if self.is_observation and self.is_dfs:
            if isinstance(self.data, types.DfsType):
                self.quantity = self.data.items[self.item_idx].type
                ds = self.data.read().to_xarray()
                _loader = parsing.get_dataset_loader(ds)
                ds = _loader(ds)
                self.is_dfs = False

        if not self.is_dfs:
            ds = parsing.rename_coords(ds)
            self.data = ds[self.additional_keys + [self.item_key]]
            if self.is_observation:
                self.data = self.data.dropna("time", how="any")

        else:
            self.quantity = self.data.items[self.item_idx].type
            if self.name is None:
                self.name = self.item_key

    def _check_field(self):
        """Maybe come up with somthing better here?"""
        target_coords = ("time", "x", "y")
        present_coords = [c for c in self.data.coords if c in target_coords]
        if len(present_coords) > 1 and self.data[
            self.item_key
        ].size == parsing._get_expected_size_if_grid(self.data[self.item_key]):
            self.is_field = True
        else:
            self.is_field = False

    def _check_point_or_track(self) -> None:
        if self.is_dfs or self.is_field:
            return

        # combine spatial variables present in data variables and coordinates
        spatial_variables = {}
        for c in ("x", "y"):
            if c in self.data.coords:
                spatial_variables[c] = self.data.coords[c]
            elif c in self.data.data_vars:
                spatial_variables[c] = self.data[c]

        if not spatial_variables:
            self.is_point, self.is_track = True, False
            return

        # The coordinates might be present, but only have one value combination, point data
        # If this is the case, we can extract the point coordinates from the data
        # and the user does not need to provide them.
        if all(np.unique(d.values).shape == (1,) for d in spatial_variables.values()):
            self.is_point, self.is_track = True, False
            self.x_point, self.y_point = (
                spatial_variables["x"].values[0],
                spatial_variables["y"].values[0],
            )

        else:
            self.is_point, self.is_track = False, True

    @staticmethod
    def check_compatibility(
        containers: list["DataContainer"],
    ) -> list[tuple[int, int]]:
        """
        Checks if the provided DataContainers are compatible for comparison.
        Implemented as a static method, so it may also be used for more complex validation
        of multiple DataContainers in any higher level collection of models and observations.
        Returns a list of tuples of the indices of compatible DataContainers.
        """
        if len(containers) < 2:
            return

        if not all(isinstance(c, DataContainer) for c in containers):
            raise TypeError(
                "All provided DataContainers must be of type DataContainer."
            )

        model_results = [(i, c) for i, c in enumerate(containers) if c.is_result]
        observations = [(i, c) for i, c in enumerate(containers) if c.is_observation]

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
        for i_m, m in model_results:
            for i_o, o in observations:
                if o.is_track and m.is_point:
                    not_ok.append((i_m, i_o))
                else:
                    ok.append((i_m, i_o))

        if not ok:
            raise ValueError(
                "No compatible model results and observations found for comparison."
            )

        if not_ok:
            for m, o in not_ok:
                logging.warning(
                    f"Can't compare track observation to point model results: {containers[m]} and {containers[o]}"
                )

        return ok

    def compare(self, other: Union["DataContainer", list["DataContainer"]]):
        if isinstance(other, list):
            return compare(other + [self])
        else:
            return compare([self, other])


def compare(data_containers: list["DataContainer"]) -> Dict[str, xr.Dataset]:
    """
    Returns a dictionary of xarray datasets with one entry for each observation.
    Each dataset contains the observation (limited to the model domains), as well as
    the model results interpolated to match the observation (in time and space).
    """
    observations = [c for c in data_containers if c.is_observation]
    dfs_results = [c for c in data_containers if c.is_result and c.is_dfs]
    xarray_results = [c for c in data_containers if c.is_result and not c.is_dfs]

    observation_extractions = {}
    if not observations or not (dfs_results or xarray_results):
        return
    for o in observations:
        if dfs_results and o.is_point_observation:
            ds = parsing.dfs_extract_point(o, dfs_results)

        elif dfs_results and o.is_track_observation:
            ds = parsing.dfs_extract_track(o, dfs_results)

        elif xarray_results and o.is_point_observation:
            ds = parsing.xarray_extract_point(o, xarray_results)

        elif xarray_results and o.is_track_observation:
            ds = parsing.xarray_extract_track(o, xarray_results)

        observation_extractions[o.name] = ds

    return observation_extractions
