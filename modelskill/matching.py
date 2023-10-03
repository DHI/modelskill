from __future__ import annotations
from datetime import timedelta
from pathlib import Path

from typing import (
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Union,
    Sequence,
    get_args,
)
import warnings
import numpy as np
import pandas as pd
import xarray as xr

import mikeio

from modelskill import ModelResult
from modelskill.timeseries import TimeSeries
from modelskill.types import GeometryType, Quantity, Period
from .model import protocols
from .model.grid import GridModelResult
from .model.dfsu import DfsuModelResult
from .model.track import TrackModelResult
from .observation import Observation, PointObservation, TrackObservation
from .comparison import Comparer, ComparerCollection
from . import __version__

TimeDeltaTypes = Union[float, int, np.timedelta64, pd.Timedelta, timedelta]
IdOrNameTypes = Optional[Union[int, str]]
# ModelResultTypes = Union[ModelResult, DfsuModelResult, str]
GeometryTypes = Optional[Literal["point", "track", "unstructured", "grid"]]
MRInputType = Union[
    str,
    Path,
    mikeio.DataArray,
    mikeio.Dataset,
    mikeio.Dfs0,
    mikeio.dfsu.Dfsu2DH,
    pd.DataFrame,
    pd.Series,
    xr.Dataset,
    xr.DataArray,
    TimeSeries,
    GridModelResult,
    DfsuModelResult,
    TrackModelResult,
]
ObsInputType = Union[
    str,
    Path,
    mikeio.DataArray,
    mikeio.Dataset,
    mikeio.Dfs0,
    pd.DataFrame,
    pd.Series,
    # protocols.Observation,
    Observation,
]


def from_matched(
    data: Union[str, Path, pd.DataFrame, mikeio.Dfs0, mikeio.Dataset],
    *,
    obs_item: str | int | None = 0,
    mod_items: Optional[Iterable[str | int]] = None,
    aux_items: Optional[Iterable[str | int]] = None,
    quantity: Optional[Quantity] = None,
    name: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    z: Optional[float] = None,
) -> Comparer:
    """Create a Comparer from observation and model results that are already matched (aligned)
    Parameters
    ----------
    data : [pd.DataFrame,str,Path,mikeio.Dfs0, mikeio.Dataset]
        DataFrame (or object that can be converted to a DataFrame e.g. dfs0)
        with columns obs_item, mod_items, aux_items
    obs_item : [str,int], optional
        Name or index of observation item, by default first item
    mod_items : Iterable[str,int], optional
        Names or indicies of model items, if None all remaining columns are model items, by default None
    aux_items : Iterable[str,int], optional
        Names or indicies of auxiliary items, by default None
    quantity : Quantity, optional
        Quantity of the observation and model results, by default Quantity(name="Undefined", unit="Undefined")
    name : str, optional
        Name of the comparer, by default None (will be set to obs_item)
    x : float, optional
        x-coordinate of observation, by default None
    y : float, optional
        y-coordinate of observation, by default None
    z : float, optional
        z-coordinate of observation, by default None
    Examples
    --------
    >>> import pandas as pd
    >>> import modelskill as ms
    >>> df = pd.DataFrame({'stn_a': [1,2,3], 'local': [1.1,2.1,3.1]}, index=pd.date_range('2010-01-01', periods=3))
    >>> cmp = ms.from_matched(df, obs_item='stn_a') # remaining columns are model results
    >>> cmp
    <Comparer>
    Quantity: Undefined [Undefined]
    Observation: stn_a, n_points=3
     Model: local, rmse=0.100
    >>> df = pd.DataFrame({'stn_a': [1,2,3], 'local': [1.1,2.1,3.1], 'global': [1.2,2.2,3.2], 'nonsense':[1,2,3]}, index=pd.date_range('2010-01-01', periods=3))
    >>> cmp = ms.from_matched(df, obs_item='stn_a', mod_items=['local', 'global'])
    >>> cmp
    <Comparer>
    Quantity: Undefined [Undefined]
    Observation: stn_a, n_points=3
        Model: local, rmse=0.100
        Model: global, rmse=0.200
    """
    # pre-process if dfs0, or mikeio.Dataset
    if isinstance(data, (str, Path)):
        assert Path(data).suffix == ".dfs0", "File must be a dfs0 file"
        data = mikeio.read(data)  # now mikeio.Dataset
    elif isinstance(data, mikeio.Dfs0):
        data = data.read()  # now mikeio.Dataset
    if isinstance(data, mikeio.Dataset):
        assert len(data.shape) == 1, "Only 0-dimensional data are supported"
        if quantity is None:
            quantity = Quantity.from_mikeio_iteminfo(data.items[obs_item])
        data = data.to_dataframe()

    cmp = Comparer.from_matched_data(
        data,
        obs_item=obs_item,
        mod_items=mod_items,
        aux_items=aux_items,
        name=name,
        x=x,
        y=y,
        z=z,
    )
    if quantity is not None:
        cmp.quantity = quantity
    return cmp


def compare(
    obs: Union[ObsInputType, Sequence[ObsInputType]],
    mod: Union[MRInputType, Sequence[MRInputType]],
    *,
    obs_item: Optional[IdOrNameTypes] = None,
    mod_item: Optional[IdOrNameTypes] = None,
    gtype: Optional[GeometryTypes] = None,
    max_model_gap=None,
) -> ComparerCollection:
    """Compare observations and model results
    Parameters
    ----------
    obs : (str, pd.DataFrame, Observation)
        Observation to be compared
    mod : (str, pd.DataFrame, ModelResultInterface)
        Model result to be compared
    obs_item : (int, str), optional
        observation item, by default None
    mod_item : (int, str), optional
        model item, by default None
    gtype : (str, optional)
        Geometry type of the model result. If not specified, it will be guessed.
    max_model_gap : (float, optional)
        Maximum gap in the model result, by default None
    Returns
    -------
    ComparerCollection
        To be used for plotting and statistics
    """
    if isinstance(obs, get_args(ObsInputType)):
        cmp = _single_obs_compare(
            obs,
            mod,
            obs_item=obs_item,
            mod_item=mod_item,
            gtype=gtype,
            max_model_gap=max_model_gap,
        )
        clist = [cmp]
    elif isinstance(obs, Sequence):
        clist = [
            _single_obs_compare(
                o,
                mod,
                obs_item=obs_item,
                mod_item=mod_item,
                gtype=gtype,
                max_model_gap=max_model_gap,
            )
            for o in obs
        ]
    else:
        raise ValueError(f"Unknown obs type {type(obs)}")

    return ComparerCollection(clist)


def _single_obs_compare(
    obs: ObsInputType,
    mod: Union[MRInputType, Sequence[MRInputType]],
    *,
    obs_item=None,
    mod_item=None,
    gtype: Optional[GeometryTypes] = None,
    max_model_gap=None,
) -> Comparer:
    """Compare a single observation with multiple models"""
    obs = _parse_single_obs(obs, obs_item, gtype=gtype)
    mod = _parse_models(mod, mod_item, gtype=gtype)
    emods = _extract_from_models(obs, mod)  # type: ignore

    raw_mod_data = parse_modeldata_list(emods)
    matched_data = match_time(obs, raw_mod_data, max_model_gap)

    return Comparer(matched_data=matched_data, raw_mod_data=raw_mod_data)


def _interp_time(df: pd.DataFrame, new_time: pd.DatetimeIndex) -> pd.DataFrame:
    """Interpolate time series to new time index"""
    new_df = (
        df.reindex(df.index.union(new_time))
        .interpolate(method="time", limit_area="inside")
        .reindex(new_time)
    )
    return new_df


def _time_delta_to_pd_timedelta(time_delta: TimeDeltaTypes) -> pd.Timedelta:
    if isinstance(time_delta, (timedelta, np.timedelta64)):
        time_delta = pd.Timedelta(time_delta)
    elif np.isscalar(time_delta):
        # assume seconds
        time_delta = pd.Timedelta(time_delta, "s")  # type: ignore
    assert isinstance(time_delta, pd.Timedelta)
    return time_delta


def _remove_model_gaps(
    df: pd.DataFrame,
    mod_index: pd.DatetimeIndex,
    max_gap: TimeDeltaTypes,
) -> pd.DataFrame:
    """Remove model gaps longer than max_gap from dataframe"""
    max_gap = _time_delta_to_pd_timedelta(max_gap)
    valid_time = _get_valid_query_time(mod_index, df.index, max_gap)
    return df.loc[valid_time]


def _get_valid_query_time(
    mod_index: pd.DatetimeIndex, obs_index: pd.DatetimeIndex, max_gap: pd.Timedelta
):
    """Used only by _remove_model_gaps"""
    # init dataframe of available timesteps and their index
    df = pd.DataFrame(index=mod_index)
    df["idx"] = range(len(df))

    # for query times get available left and right index of source times
    df = _interp_time(df, obs_index).dropna()
    df["idxa"] = np.floor(df.idx).astype(int)
    df["idxb"] = np.ceil(df.idx).astype(int)

    # time of left and right source times and time delta
    df["ta"] = mod_index[df.idxa]
    df["tb"] = mod_index[df.idxb]
    df["dt"] = df.tb - df.ta

    # valid query times where time delta is less than max_gap
    valid_idx = df.dt <= max_gap
    return valid_idx


def _mask_model_outside_observation_track(name, df_mod, df_obs) -> None:
    if len(df_mod) == 0:
        return
    if len(df_mod) != len(df_obs):
        raise ValueError("model and observation data must have same length")

    mod_xy = df_mod[["x", "y"]]
    obs_xy = df_obs[["x", "y"]]
    d_xy = np.sqrt(np.sum((obs_xy - mod_xy) ** 2, axis=1))

    # Find minimal accepted distance from data
    # (could be long/lat or x/y, small or large domain)
    tol_xy = _minimal_accepted_distance(obs_xy)
    mask = d_xy > tol_xy
    df_mod.loc[mask, name] = np.nan
    if all(mask):
        warnings.warn("no (spatial) overlap between model and observation points")


def _get_global_start_end(idxs: Iterable[pd.DatetimeIndex]) -> Period:
    starts = [x[0] for x in idxs if len(x) > 0]
    ends = [x[-1] for x in idxs if len(x) > 0]

    if len(starts) == 0:
        return Period(start=None, end=None)

    return Period(start=min(starts), end=max(ends))


def match_time(
    observation: Observation,
    raw_mod_data: Dict[str, pd.DataFrame],
    max_model_gap: Optional[TimeDeltaTypes] = None,
) -> xr.Dataset:
    """Match observation with one or more model results in time domain
    and return as xr.Dataset in the format used by modelskill.Comparer

    Will interpolate model results to observation time.

    Note: assumes that observation and model data are already matched in space.

    Parameters
    ----------
    observation : Observation
        Observation to be matched
    raw_mod_data : Dict[str, pd.DataFrame]
        Dictionary of model results ready for interpolation
    max_model_gap : Optional[TimeDeltaTypes], optional
        In case of non-equidistant model results (e.g. event data),
        max_model_gap can be given e.g. as seconds, by default None

    Returns
    -------
    xr.Dataset
        Matched data in the format used by modelskill.Comparer
    """
    obs_name = "Observation"
    mod_names = list(raw_mod_data.keys())
    idxs = [m.index for m in raw_mod_data.values()]
    period = _get_global_start_end(idxs)

    assert isinstance(observation, (PointObservation, TrackObservation))
    gtype = "point" if isinstance(observation, PointObservation) else "track"
    observation = observation.copy()
    observation.trim(period.start, period.end)

    first = True
    for name, mdata in raw_mod_data.items():
        df = _model2obs_interp(observation, mdata, max_model_gap)
        if gtype == "track":
            # TODO why is it necessary to do mask here? Isn't it an error if the model data is outside the observation track?
            df_obs = observation.data.to_pandas()  # TODO: xr.Dataset
            _mask_model_outside_observation_track(name, df, df_obs)

        if first:
            data = df
        else:
            data[name] = df[name]

        first = False

    data.index.name = "time"
    data = data.dropna()
    data = data.to_xarray()
    data.attrs["gtype"] = str(gtype)

    if gtype == "point":
        data["x"] = observation.x
        data["y"] = observation.y
        data["z"] = observation.z  # type: ignore

    data.attrs["name"] = observation.name
    data.attrs["quantity_name"] = observation.quantity.name
    data["x"].attrs["kind"] = "position"
    data["y"].attrs["kind"] = "position"
    data[obs_name].attrs["kind"] = "observation"
    data[obs_name].attrs["unit"] = observation.quantity.unit
    data[obs_name].attrs["color"] = observation.color
    data[obs_name].attrs["weight"] = observation.weight
    for n in mod_names:
        data[n].attrs["kind"] = "model"

    data.attrs["modelskill_version"] = __version__

    return data


def _model2obs_interp(
    obs, mod_df: pd.DataFrame, max_model_gap: Optional[TimeDeltaTypes]
) -> pd.DataFrame:
    """interpolate model to measurement time"""
    _obs_name = "Observation"
    df = _interp_time(mod_df.dropna(), obs.time)
    df[_obs_name] = obs.values

    if max_model_gap is not None:
        df = _remove_model_gaps(df, mod_df.dropna().index, max_model_gap)

    return df


def _minimal_accepted_distance(obs_xy):
    # all consequtive distances
    vec = np.sqrt(np.sum(np.diff(obs_xy, axis=0), axis=1) ** 2)
    # fraction of small quantile
    return 0.5 * np.quantile(vec, 0.1)


def parse_modeldata_list(modeldata) -> Dict[str, pd.DataFrame]:
    """Convert to dict of dataframes"""
    if not isinstance(modeldata, Sequence):
        modeldata = [modeldata]

    mod_dfs = [_parse_single_modeldata(m) for m in modeldata]
    return {m.columns[-1]: m for m in mod_dfs if m is not None}


def _parse_single_modeldata(modeldata) -> pd.DataFrame:
    """Convert to dataframe and set index to pd.DatetimeIndex"""
    if hasattr(modeldata, "to_dataframe"):
        mod_df = modeldata.to_dataframe()
    elif isinstance(modeldata, pd.DataFrame):
        mod_df = modeldata
    else:
        raise ValueError(
            f"Unknown modeldata type '{type(modeldata)}' (mikeio.Dataset, xr.DataArray, xr.Dataset or pd.DataFrame)"
        )

    if not isinstance(mod_df.index, pd.DatetimeIndex):
        raise ValueError(
            "Modeldata index must be datetime-like (pd.DatetimeIndex, pd.to_datetime)"
        )

    time = mod_df.index.round(freq="100us")  # 0.0001s accuracy
    mod_df.index = pd.DatetimeIndex(time, freq="infer")
    return mod_df


def _parse_single_obs(
    obs, item=None, gtype: Optional[GeometryTypes] = None
) -> Observation:
    if isinstance(obs, Observation):
        if item is not None:
            raise ValueError(
                "obs_item argument not allowed if obs is an modelskill.Observation type"
            )
        return obs
    else:
        if (gtype is not None) and (
            GeometryType.from_string(gtype) == GeometryType.TRACK
        ):
            return TrackObservation(obs, item=item)
        else:
            return PointObservation(obs, item=item)


def _parse_models(
    mod, item: Optional[IdOrNameTypes] = None, gtype: Optional[GeometryTypes] = None
):
    """Return a list of ModelResult objects"""
    if isinstance(mod, get_args(MRInputType)):
        return [_parse_single_model(mod, item=item, gtype=gtype)]
    elif isinstance(mod, Sequence):
        return [_parse_single_model(m, item=item, gtype=gtype) for m in mod]
    else:
        raise ValueError(f"Unknown mod type {type(mod)}")


def _parse_single_model(
    mod, item: Optional[IdOrNameTypes] = None, gtype: Optional[GeometryTypes] = None
):
    if isinstance(mod, protocols.ModelResult):
        if item is not None:
            raise ValueError(
                "mod_item argument not allowed if mod is an modelskill.ModelResult"
            )
        return mod

    try:
        return ModelResult(mod, item=item, gtype=gtype)
    except ValueError as e:
        raise ValueError(
            f"Could not compare. Unknown model result type {type(mod)}. {str(e)}"
        )


def _extract_from_models(obs, mods: List[protocols.ModelResult]) -> List[pd.DataFrame]:
    df_model = []
    for m in mods:
        mr: TimeSeries = m.extract(obs) if hasattr(m, "extract") else m

        # TODO: temporary solution until complete swich to xr.Dataset
        # mr.data if isinstance(mr.data, pd.DataFrame) else
        df = mr.to_dataframe()

        # TODO is this robust enough?
        old_item = df.columns.values[-1]  # TODO: xr.Dataset
        df = df.rename(columns={old_item: mr.name})  # TODO: xr.Dataset
        if (df is not None) and (len(df) > 0):  # TODO: xr.Dataset
            df_model.append(df)
        else:
            warnings.warn(
                f"No data found when extracting '{obs.name}' from model '{mr.name}'"
            )
    return df_model
