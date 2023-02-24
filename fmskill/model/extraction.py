import pandas as pd
import xarray as xr

from fmskill.model import protocols, PointModelResult, TrackModelResult
from fmskill.observation import PointObservation, TrackObservation

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


def rename_coords_xr(ds: xr.Dataset) -> xr.Dataset:
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


def rename_coords_pd(df: pd.DataFrame) -> pd.DataFrame:
    """Rename coordinates to standard names"""
    _mapping = {
        c: TIME_COORDINATE_NAME_MAPPING[c.lower()]
        for c in df.columns
        if c.lower() in TIME_COORDINATE_NAME_MAPPING.keys()
    }
    _mapping.update(
        {
            c: POS_COORDINATE_NAME_MAPPING[c.lower()]
            for c in df.columns
            if c.lower() in POS_COORDINATE_NAME_MAPPING.keys()
        }
    )
    return df.rename(columns=_mapping)


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
    renamed_mr_data = rename_coords_xr(mr.data)
    da = renamed_mr_data[mr.item].interp(coords=dict(x=x, y=y), method="nearest")
    df = da.to_dataframe().drop(columns=["x", "y"])
    df = df.rename(columns={df.columns[-1]: mr.name})

    return PointModelResult(
        data=df.dropna(),
        x=da.x.item(),
        y=da.y.item(),
        item=mr.item,
        name=mr.name,
        quantity=mr.quantity,
    )


def track_obs_from_xr_mr(
    mr: protocols.Extractable, observation: TrackObservation
) -> TrackModelResult:
    """Extract a TrackModelResult from a GridModelResult (when data is a xarray.Dataset),
    given a TrackObservation."""

    renamed_obs_data = rename_coords_pd(observation.data)
    renamed_mr_data = rename_coords_xr(mr.data)
    t = xr.DataArray(renamed_obs_data.index, dims="track")
    x = xr.DataArray(renamed_obs_data.x, dims="track")
    y = xr.DataArray(renamed_obs_data.y, dims="track")
    da = renamed_mr_data[mr.item].interp(coords=dict(time=t, x=x, y=y), method="linear")
    df = da.to_dataframe().drop(columns=["time"])
    df.index.name = "time"
    df = df.rename(columns={df.columns[-1]: mr.name})

    return TrackModelResult(
        data=df.dropna(),
        item=mr.item,
        name=mr.name,
        quantity=mr.quantity,
    )

    return df.dropna()
