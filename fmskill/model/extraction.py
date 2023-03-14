import mikeio
import numpy as np
import pandas as pd
import xarray as xr

from fmskill import utils
from fmskill.model import PointModelResult, TrackModelResult, protocols
from fmskill.observation import PointObservation, TrackObservation


def _xy_in_xr_domain(data: xr.Dataset, x: float, y: float) -> bool:
    if (x is None) or (y is None):
        raise ValueError(f"Cannot determine if point ({x}, {y}) is inside domain!")
    xmin = data.x.values.min()
    xmax = data.x.values.max()
    ymin = data.y.values.min()
    ymax = data.y.values.max()
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def _any_obs_in_model_time(
    time_obs: pd.DatetimeIndex, time_model: pd.DatetimeIndex
) -> bool:
    """Check if any observation times are in model time range"""
    return (time_obs[-1] >= time_model[0]) & (time_obs[0] <= time_model[-1])


def _validate_any_obs_in_model_time(
    obs_name: str, time_obs: pd.DatetimeIndex, time_model: pd.DatetimeIndex
) -> None:
    """Check if any observation times are in model time range"""
    ok = _any_obs_in_model_time(time_obs, time_model)
    if not ok:
        raise ValueError(
            f"Observation '{obs_name}' outside model time range! "
            + f"({time_obs[0]} - {time_obs[-1]}) not in ({time_model[0]} - {time_model[-1]})"
        )


def point_obs_from_xr_mr(
    mr: protocols.Extractable, observation: PointObservation
) -> PointModelResult:
    """Extract a PointModelResult from a GridModelResult (when data is a xarray.Dataset),
    given a PointObservation."""

    x, y = observation.x, observation.y
    if (x is None) or (y is None):
        raise ValueError(
            f"PointObservation '{observation.name}' cannot be used for extraction "
            + f"because it has None position x={x}, y={y}. Please provide position "
            + "when creating PointObservation."
        )
    if not _xy_in_xr_domain(mr.data, x, y):
        raise ValueError(
            f"PointObservation '{observation.name}' ({x}, {y}) is outside model domain!"
        )

    _validate_any_obs_in_model_time(observation.name, observation.data.index, mr.time)

    da = mr.data[mr.item].interp(coords=dict(x=x, y=y), method="nearest")
    df = da.to_dataframe().drop(columns=["x", "y"])
    df = df.rename(columns={df.columns[-1]: mr.name})

    return PointModelResult(
        data=df.dropna(),
        x=da.x.item(),
        y=da.y.item(),
        item=mr.name,
        itemInfo=mr.itemInfo,
        name=mr.name,
        quantity=mr.quantity,
    )


def extract_track_from_xr(
    mr: protocols.Extractable, observation: TrackObservation
) -> TrackModelResult:
    """Extract a TrackModelResult from a GridModelResult (when data is a xarray.Dataset),
    given a TrackObservation."""

    _validate_any_obs_in_model_time(observation.name, observation.data.index, mr.time)

    renamed_obs_data = utils.rename_coords_pd(observation.data)
    t = xr.DataArray(renamed_obs_data.index, dims="track")
    x = xr.DataArray(renamed_obs_data.x, dims="track")
    y = xr.DataArray(renamed_obs_data.y, dims="track")
    da = mr.data[mr.item].interp(coords=dict(time=t, x=x, y=y), method="linear")
    df = da.to_dataframe().drop(columns=["time"])
    # df.index.name = "time"
    df = df.rename(columns={df.columns[-1]: mr.name})

    return TrackModelResult(
        data=df.dropna(),
        item=mr.name,
        itemInfo=mr.itemInfo,
        name=mr.name,
        quantity=mr.quantity,
    )


def extract_point_from_dfsu(
    mr: protocols.Extractable, observation: PointObservation
) -> PointModelResult:
    """Extract a PointModelResult from a DfsuModelResult (when data is a Dfsu object),
    given a PointObservation."""

    assert isinstance(mr.data, (mikeio.dfsu.Dfsu2DH, mikeio.DataArray, mikeio.Dataset))

    x, y = observation.x, observation.y
    if not mr.data.geometry.contains([x, y]):
        raise ValueError(
            f"PointObservation '{observation.name}' ({x}, {y}) outside model domain!"
        )

    _validate_any_obs_in_model_time(observation.name, observation.data.index, mr.time)

    # TODO: interp2d
    xy = np.atleast_2d([x, y])
    elemids = mr.data.geometry.find_index(coords=xy)
    if isinstance(mr.data, mikeio.dfsu.Dfsu2DH):
        ds_model = mr.data.read(elements=elemids, items=[mr.item])
    elif isinstance(mr.data, mikeio.Dataset):
        ds_model = mr.data.isel(element=elemids)
    elif isinstance(mr.data, mikeio.DataArray):
        da = mr.data.isel(element=elemids)
        ds_model = mikeio.Dataset({da.name: da})
    ds_model.rename({ds_model.items[0].name: mr.name}, inplace=True)

    return PointModelResult(
        data=ds_model,
        x=ds_model.geometry.x,
        y=ds_model.geometry.y,
        item=mr.name,
        itemInfo=mr.itemInfo,
        name=mr.name,
        quantity=mr.quantity,
    )


def extract_track_from_dfsu(
    mr: protocols.Extractable, observation: TrackObservation
) -> TrackModelResult:
    """Extract a TrackModelResult from a DfsuModelResult (when data is a Dfsu object),
    given a TrackObservation."""

    assert isinstance(mr.data, (mikeio.dfsu.Dfsu2DH, mikeio.DataArray, mikeio.Dataset))

    _validate_any_obs_in_model_time(observation.name, observation.data.index, mr.time)

    if isinstance(mr.data, mikeio.dfsu.Dfsu2DH):
        ds_model = mr.data.extract_track(track=observation.data, items=[mr.item])
    elif isinstance(mr.data, (mikeio.Dataset, mikeio.DataArray)):
        ds_model = mr.data.extract_track(track=observation.data)
    ds_model.rename({ds_model.items[-1].name: mr.name}, inplace=True)

    return TrackModelResult(
        data=ds_model.dropna(),  # .to_dataframe().dropna(),
        item=mr.name,
        itemInfo=mr.itemInfo,
        name=mr.name,
        quantity=mr.quantity,
    )
