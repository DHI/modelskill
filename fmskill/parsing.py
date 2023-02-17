from inspect import getmembers, isfunction
from pathlib import Path, PosixPath
from typing import Callable, Iterable, Union

import mikeio
import numpy as np
import pandas as pd
import xarray as xr

from fmskill import metrics as mtr, types

from .utils import _as_path, make_unique_index

POS_COORDINATE_NAME_MAPPING = {
    "lon": "x",
    "longitude": "x",
    "lat": "y",
    "latitude": "y",
    "east": "x",
    "north": "y",
}
TIME_COORDINATE_NAME_MAPPING = {
    "t": "time",
    "date": "time",
}


def dfs_extract_point(observation, model_results) -> xr.Dataset:
    from fmskill.data_container import DataContainer

    if not isinstance(model_results, list):
        assert isinstance(model_results, DataContainer)
        model_results = [model_results]

    model_names = [mr.name for mr in model_results]

    _extracted_mrs = []
    attrs = {"x": {}, "y": {}}
    for mr in model_results:
        if "dfsu" in mr.file_extension:
            _assert_valid_point_obs(observation)
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
    temporally_cut_obs = fit_to_time(
        observation.data.rename({observation.item_key: "Observation"}), _extracted_mrs
    )
    ds = xr.merge([temporally_cut_obs, _extracted_mrs], join="left")

    ds = ds.interpolate_na(dim="time")
    ds = _add_source_dim(ds[model_names + ["Observation"]], retun_as_dataset=True)
    ds.attrs.update(attrs)
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

    model_names = [mr.name for mr in model_results]

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
    temporally_cut_obs = fit_to_time(
        observation.data.rename({observation.item_key: "Observation"}), _extracted_mrs
    )
    # Also need to filter the observation to the same spatial extent as the MR
    spatially_cut_obs = fit_to_spatial_extend(temporally_cut_obs, _extracted_mrs)

    # Left join the observation and the MRs
    ds = xr.merge([spatially_cut_obs, _extracted_mrs], join="left")

    # Interpolate the NaNs in the MRs
    ds = ds.interpolate_na(dim="time")
    _multi_dim = _add_source_dim(
        ds[model_names + ["Observation"]], retun_as_dataset=False
    )

    ds = ds[["x", "y"]]
    ds["variable"] = _multi_dim

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


def _assert_valid_point_obs(observation):
    if (observation.x is None) or (observation.y is None):
        raise ValueError("Missing x and/or y coordinates for point observation")


def _add_source_dim(ds: xr.Dataset, retun_as_dataset: bool) -> xr.Dataset:
    da = ds.to_array(dim="source")
    if not retun_as_dataset:
        return da
    else:
        return da.to_dataset(name="variable")


def fit_to_time(to_fit, target, time_name="time"):
    if isinstance(to_fit, xr.Dataset):
        return to_fit.sel(
            **{time_name: slice(target[time_name].min(), target[time_name].max())}
        )
    raise NotImplementedError(f"fit_to_time not implemented for type {type(to_fit)}")


def fit_to_spatial_extend(to_fit, target, x_name="x", y_name="y"):
    if isinstance(to_fit, xr.Dataset):
        return to_fit.where(
            (to_fit[x_name] >= target[x_name].min())
            & (to_fit[x_name] <= target[x_name].max())
            & (to_fit[y_name] >= target[y_name].min())
            & (to_fit[y_name] <= target[y_name].max()),
            drop=True,
        )
    raise NotImplementedError(
        f"fit_to_spatial_extend not implemented for type {type(to_fit)}"
    )


def _point_coords_to_attr(ds: xr.Dataset):
    # for point observations, we don't need to store the coordinates
    # as coordinates or data variables, we just put the extracted position
    # in the metadata
    for c in ("x", "y"):
        if c in ds.coords:
            ds.attrs[c] = ds[c].item()
            ds = ds.reset_coords(c, drop=True)
    return ds


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
        return dfs


def parse_metric(metric, return_list=False):
    if isinstance(metric, str):
        valid_metrics = [x[0] for x in getmembers(mtr, isfunction) if x[0][0] != "_"]

        if metric.lower() in valid_metrics:
            metric = getattr(mtr, metric.lower())
        else:
            raise ValueError(
                f"Invalid metric: {metric}. Valid metrics are {valid_metrics}."
            )
    elif isinstance(metric, Iterable):
        metrics = [parse_metric(m) for m in metric]
        return metrics
    elif not callable(metric):
        raise TypeError(f"Invalid metric: {metric}. Must be either string or callable.")
    if return_list:
        if callable(metric) or isinstance(metric, str):
            metric = [metric]
    return metric


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

    return non_dfs_loading_types_mapping.get(type(data))


def get_dfs_loader(
    data: types.DataInputType,
) -> types.DfsType:
    return lazy_loading_types_mapping.get(type(data))


non_dfs_loading_types_mapping = {
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
