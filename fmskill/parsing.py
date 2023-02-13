from pathlib import Path, PosixPath
from typing import Callable, Union

import mikeio
import pandas as pd
import xarray as xr
import numpy as np

from fmskill import types

from .utils import _as_path, make_unique_index

POS_COORDINATE_NAME_MAPPING = {
    # "x": "x",
    # "y": "y",
    "lon": "x",
    "longitude": "x",
    "lat": "y",
    "latitude": "y",
    "east": "x",
    "north": "y",
}
TIME_COORDINATE_NAME_MAPPING = {
    # "time": "time",
    "t": "time",
    "date": "time",
}


def dfs_extract_point(observation, model_results) -> xr.Dataset:
    _extracted_mrs = []
    attrs = {"x": {}, "y": {}}
    for mr in model_results:
        if "dfsu" in mr.file_extension:
            if (observation.x is None) or (observation.y is None):
                raise ValueError(
                    f"PointObservation '{observation.name}' cannot be used for extraction "
                    + f"because it has None position x={observation.x}, y={observation.y}. "
                    + "Please provide position when creating PointObservation."
                )
            _extr = _extract_point_dfsu(
                dfsu=mr.data, x=observation.x, y=observation.y, item=mr.item_idx
            )
        elif "dfs0" in mr.file_extension:
            _extr = _extract_point_dfs0(dfs0=mr.data, item=mr.item_idx)

        _extr = rename_coords(_extr).rename({mr.item_key: mr.name})
        if "x" in _extr.attrs:
            attrs["x"][mr.name] = _extr.attrs.pop("x")
            attrs["y"][mr.name] = _extr.attrs.pop("y")
        _extracted_mrs.append(_extr)

    attrs["x"][observation.name] = observation.x
    attrs["y"][observation.name] = observation.y

    _extracted_mrs = xr.merge(_extracted_mrs, combine_attrs="no_conflicts")
    temporally_cut_obs = observation.data.sel(
        time=slice(_extracted_mrs.time.min(), _extracted_mrs.time.max())
    )
    temporally_cut_obs.attrs.update(attrs)
    ds = xr.merge([temporally_cut_obs, _extracted_mrs], join="left")

    ds = ds.interpolate_na(dim="time")

    return ds


def _point_coords_to_attr(ds: xr.Dataset):
    # for point observations, we don't need to store the coordinates
    # as coordinates or data variables, we just put the extracted position
    # in the metadata
    for c in ("x", "y"):
        if c in ds.coords:
            ds.attrs[c] = ds[c].item()
            ds = ds.reset_coords(c, drop=True)
    return ds


def _extract_point_dfsu(dfsu: mikeio.dfsu._Dfsu, x, y, item) -> xr.Dataset:
    xy = np.atleast_2d([x, y])
    elemids = dfsu.geometry.find_index(coords=xy)
    ds_model: mikeio.Dataset = dfsu.read(elements=elemids, items=[item])
    ds: xr.Dataset = ds_model.to_xarray()

    ds = _point_coords_to_attr(ds)
    return ds


def _extract_point_dfs0(dfs0: mikeio.Dfs0, item) -> xr.Dataset:
    ds_model = dfs0.read(items=[item])
    ds = ds_model.to_xarray()
    return _point_coords_to_attr(ds)


def dfs_extract_track(observation, model_results):
    from fmskill.data_container import DataContainer

    if not isinstance(model_results, list):
        assert isinstance(model_results, DataContainer)
        model_results = [model_results]

    _extracted_mrs = []
    for mr in model_results:
        # Dfs extraction returns spatial and temporal intersection of MR and observation,
        # but still containing the original coordinates of the observation. Variables that
        # are outside the MR domain are filled with NaNs.
        if "dfsu" in mr.file_extension:
            _extr = _extract_track_dfsu(
                dfsu=mr.data, observation=observation, item=mr.item_idx
            )
        elif "dfs0" in mr.file_extension:
            _extr = _extract_track_dfs0(dfs0=mr.data, item=mr.item_idx)

        # Drop the NaNs, as we do not want to extrapolate
        _extr = _extr.dropna(dim="time")
        _extr = rename_coords(_extr).rename({mr.item_key: mr.name})

        _extracted_mrs.append(_extr)

    # Merge the individually extracted MRs into one dataset
    _extracted_mrs = xr.merge(_extracted_mrs)

    # Cut the observation to the same time period as the MR
    temporally_cut_obs = observation.data.sel(
        time=slice(_extracted_mrs.time.min(), _extracted_mrs.time.max())
    )
    # Also need to filter the observation to the same spatial extent as the MR
    spatially_cut_obs = temporally_cut_obs.where(
        (temporally_cut_obs.x >= _extracted_mrs.x.min())
        & (temporally_cut_obs.x <= _extracted_mrs.x.max())
        & (temporally_cut_obs.y >= _extracted_mrs.y.min())
        & (temporally_cut_obs.y <= _extracted_mrs.y.max()),
        drop=True,
    )

    # Left join the observation and the MRs
    ds = xr.merge([spatially_cut_obs, _extracted_mrs], join="left")

    # Interpolate the NaNs in the MRs
    ds = ds.interpolate_na(dim="time")

    return ds


def _extract_track_dfsu(dfsu: mikeio.dfsu._Dfsu, observation, item: int) -> xr.Dataset:
    ds_model = dfsu.extract_track(track=observation.data.to_dataframe(), items=[item])
    return ds_model.to_xarray()


def _extract_track_dfs0(dfs0: mikeio.Dfs0, item: int) -> xr.Dataset:
    ds_model = dfs0.read(items=[0, 1, item])
    df = ds_model.to_dataframe().dropna()
    df.index = make_unique_index(df.index, offset_duplicates=0.001)
    if isinstance(df.index, pd.DatetimeIndex):
        df.index.name = "time"
    return df.to_xarray()


def xarray_extract_point(result, observation) -> xr.Dataset:
    return result.data.interp(
        coords=dict(x=observation.x, y=observation.y), method="nearest"
    )


def xarray_extract_track(result, observation) -> xr.Dataset:
    return result.data.interp(
        x=observation.x, y=observation.y, time=observation.data.time, method="linear"
    ).reset_coords(["x", "y"])


def ds_from_filepath(filepath: Union[str, Path, list]):
    """Get a DataSet from a filepath. Does not support dfs files."""
    filename = _as_path(filepath)
    ext = filename.suffix
    if "dfs" not in ext:
        if "*" not in str(filename):
            return xr.open_dataset(filename)
        elif isinstance(filepath, str) or isinstance(filepath, list):
            return xr.open_mfdataset(filepath)


def dfs_from_filepath(filepath: Union[str, Path, list]) -> types.DfsType:
    """
    Return a lazy loading object for a filepath.
    Currently supported formats: .dfs0, .dfsu
    """
    filename = _as_path(filepath)
    ext = filename.suffix
    if "dfs" in ext:
        dfs = mikeio.open(filename)
        # item = _dfs_get_item_index(dfs, item)
        return dfs


# def array_from_pd_series(
#     series: pd.Series, *args, **kwargs
# ) -> Tuple[xr.DataArray, str]:
#     """Get a DataArray from a pandas Series."""
#     return series.to_xarray(), series.name


def array_from_pd_dataframe(df: pd.DataFrame, *args, **kwargs) -> xr.Dataset:
    """Get a DataSet from a pandas DataFrame."""

    df.index = make_unique_index(pd.to_datetime(df.index), offset_duplicates=0.001)
    return df.to_xarray()


def get_dataset_loader(
    data: types.DataInputType,
) -> Callable[[types.DataInputType], xr.Dataset]:
    """Check if the provided data can be loaded eagerly. If so, return the loader function,
    otherwise return None."""

    if isinstance(data, (str, Path)):
        _data = _as_path(data)
        if "dfs" in _data.suffix:
            # dfs are loaded lazily, so return None here
            return None

    return eager_loading_types_mapping.get(type(data))


def get_dfs_loader(
    data: types.DataInputType,
) -> types.DfsType:
    return lazy_loading_types_mapping.get(type(data))


eager_loading_types_mapping = {
    xr.DataArray: lambda x: x.to_dataset(),
    xr.Dataset: lambda x: x,
    str: ds_from_filepath,
    Path: ds_from_filepath,
    PosixPath: ds_from_filepath,
    list: ds_from_filepath,
    # pd.Series: array_from_pd_series,
    pd.DataFrame: array_from_pd_dataframe,
}

lazy_loading_types_mapping = {
    str: dfs_from_filepath,
    Path: dfs_from_filepath,
    PosixPath: dfs_from_filepath,
}


def rename_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename coordinates to standard names"""
    ds = ds.rename(
        {
            c: TIME_COORDINATE_NAME_MAPPING[c.lower()]
            for c in list(ds.coords) + list(ds.data_vars)
            if c.lower() in TIME_COORDINATE_NAME_MAPPING.keys()
        }
    )
    ds = ds.rename(
        {
            c: POS_COORDINATE_NAME_MAPPING[c.lower()]
            for c in list(ds.coords) + list(ds.data_vars)
            if c.lower() in POS_COORDINATE_NAME_MAPPING.keys()
        }
    )
    return ds


def _get_expected_size_if_grid(da: xr.DataArray) -> int:
    """Returns the expected size of a grid data array"""
    _size = 1
    for c in [d for d in da.coords if d in ("time", "x", "y")]:
        _size *= da.coords[c].size
    return _size


def get_item_name_xr_ds(ds: xr.Dataset, item, item_names=None) -> tuple[str, int]:
    """Returns the name and index of the requested data variable, provided
    either as either a str or int."""
    if item_names is None:
        item_names = list(ds.data_vars)
    n_items = len(item_names)
    if item is None:
        if n_items == 1:
            return item_names[0], 0
        else:
            raise ValueError(
                f"item must be specified when more than one item available. Available items: {item_names}"
            )
    if isinstance(item, int):
        if item < 0:  # Handle negative indices
            item = n_items + item
        if (item < 0) or (item >= n_items):
            raise IndexError(f"item {item} out of range (0, {n_items-1})")
        return item_names[item], item
    elif isinstance(item, str):
        if item not in item_names:
            raise KeyError(f"item must be one of {item_names}")
        return item, item_names.index(item)
    else:
        raise TypeError("item must be int or string")


def get_item_name_dfs(dfs: types.DfsType, item) -> tuple[str, int]:
    """Returns the name and index of the requested variable, provided
    either as either a str or int."""
    item_names = [i.name for i in dfs.items]
    n_items = len(item_names)
    if item is None:
        if n_items == 1:
            return item_names[0], 0
        else:
            raise ValueError(
                f"item must be specified when more than one item available. Available items: {item_names}"
            )
    if isinstance(item, int):
        if item < 0:  # Handle negative indices
            item = n_items + item
        if (item < 0) or (item >= n_items):
            raise IndexError(f"item {item} out of range (0, {n_items-1})")
        return item_names[item], item
    elif isinstance(item, str):
        if item not in item_names:
            raise KeyError(f"item must be one of {item_names}")
        return item, item_names.index(item)
    else:
        raise TypeError("item must be int or string")


def get_coords_in_data_vars(ds: Union[xr.Dataset, types.DfsType]) -> list[str]:
    """Returns a list of coordinate names that are also data variables"""
    if isinstance(ds, xr.Dataset):
        return [c for c in ds.data_vars if c in ("x", "y", "time")]
    elif isinstance(ds, (mikeio.Dfs0, mikeio.Dfsu)):
        return [c for c in ds.items if c in ("x", "y", "time")]


def _dfs_get_item_index(dfs, item):
    available_names = [i.name for i in dfs.items]
    lower_case_names = [i.lower() for i in available_names]
    if item is None:
        if len(dfs.items) > 1:
            raise ValueError(
                f"Found more than one item in dfs. Please specify item. Available: {available_names}"
            )
        else:
            return 0
    elif isinstance(item, str):
        if item.lower() not in lower_case_names:
            raise ValueError(
                f"Requested item {item} not found in dfs file. Available: {available_names}"
            )
        return lower_case_names.index(item.lower())

    elif isinstance(item, int):
        idx = item
        n_items = len(dfs.items)
        if idx < 0:  # Handle negative indices
            idx = n_items + idx
        if (idx < 0) or (idx >= n_items):
            raise IndexError(f"item {item} out of range (0, {n_items-1})")

        return idx


def validate_and_format_xarray(da: xr.DataArray):
    new_names = {}
    coords = da.coords
    for coord in coords:
        c = coord.lower()
        if ("x" not in new_names) and (("lon" in c) or ("east" in c)):
            new_names[coord] = "x"
        elif ("y" not in new_names) and (("lat" in c) or ("north" in c)):
            new_names[coord] = "y"
        elif ("time" not in new_names) and (("time" in c) or ("date" in c)):
            new_names[coord] = "time"

    if len(new_names) > 0:
        da = da.rename(new_names)

    cnames = list(da.coords)
    for c in ["x", "y", "time"]:
        if c not in cnames:
            raise ValueError(f"{c} not found in coords {cnames}")

    if not isinstance(coords["time"].to_index(), pd.DatetimeIndex):
        raise ValueError(f"Time coordinate is not equivalent to DatetimeIndex")

    return da


def validate_input_data(data, item) -> None:
    """Validates the input data to ensure that a loader will available for the provided data."""

    if isinstance(data, mikeio.Dataset):
        raise ValueError("mikeio.Dataset not supported, but mikeio.DataArray is")

    if not isinstance(
        data,
        (
            str,
            Path,
            list,
            mikeio.DataArray,
            pd.DataFrame,
            pd.Series,
            xr.Dataset,
            xr.DataArray,
        ),
    ):
        # if not isinstance(data, types.DataInputType):
        raise ValueError(
            "Input type not supported (str, Path, mikeio.DataArray, DataFrame, xr.DataArray)"
        )
    if not isinstance(item, types.ItemSpecifier):
        raise ValueError("Invalid type for item argument (int, str, None)")
