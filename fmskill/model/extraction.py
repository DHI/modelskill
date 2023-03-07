import mikeio
import numpy as np
import xarray as xr

from fmskill import utils
from fmskill.model import PointModelResult, TrackModelResult, protocols
from fmskill.observation import PointObservation, TrackObservation


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


def track_obs_from_xr_mr(
    mr: protocols.Extractable, observation: TrackObservation
) -> TrackModelResult:
    """Extract a TrackModelResult from a GridModelResult (when data is a xarray.Dataset),
    given a TrackObservation."""

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
        item=mr.item,
        itemInfo=mr.itemInfo,
        name=mr.name,
        quantity=mr.quantity,
    )


def point_obs_from_dfsu_mr(
    mr: protocols.Extractable, observation: PointObservation
) -> PointModelResult:
    """Extract a PointModelResult from a DfsuModelResult (when data is a Dfsu object),
    given a PointObservation."""

    xy = np.atleast_2d([observation.x, observation.y])
    elemids = mr.data.geometry.find_index(coords=xy)
    ds_model = mr.data.read(elements=elemids, items=[mr.item])
    ds_model.rename({ds_model.items[0].name: mr.name}, inplace=True)

    df = ds_model.to_dataframe().dropna()
    df.index = utils.make_unique_index(df.index, offset_duplicates=0.001)

    return PointModelResult(
        data=df,
        x=ds_model.geometry.x,
        y=ds_model.geometry.y,
        item=mr.name,
        itemInfo=mr.itemInfo,
        name=mr.name,
        quantity=mr.quantity,
    )


def track_obs_from_dfsu_mr(
    mr: protocols.Extractable, observation: TrackObservation
) -> TrackModelResult:
    """Extract a TrackModelResult from a DfsuModelResult (when data is a Dfsu object),
    given a TrackObservation."""

    ds_model = mr.data.extract_track(track=observation.data, items=[mr.item])
    ds_model.rename({ds_model.items[-1].name: mr.name}, inplace=True)

    return TrackModelResult(
        data=ds_model.to_dataframe().dropna(),
        item=mr.item,
        itemInfo=mr.itemInfo,
        name=mr.name,
        quantity=mr.quantity,
    )


def point_obs_from_mike_ds_mr(
    mr: protocols.Extractable, observation: PointObservation
) -> PointModelResult:

    dap = mr.data.sel(x=observation.x, y=observation.y)
    dap.name = mr.name
    # ds_model.rename({ds_model.items[0].name: self.name}, inplace=True)

    return PointModelResult(
        data=mikeio.Dataset(dap).to_dataframe().dropna(),
        x=dap.x.item(),
        y=dap.y.item(),
        item=mr.item,
        name=mr.name,
        quantity=mr.quantity,
    )
