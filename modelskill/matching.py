from __future__ import annotations
from datetime import timedelta
from pathlib import Path
import warnings

from typing import (
    Dict,
    Iterable,
    Collection,
    List,
    Literal,
    Optional,
    Union,
    Sequence,
    get_args,
    TypeVar,
    Any,
    overload,
)
import numpy as np
import pandas as pd
import xarray as xr

import mikeio


from . import model_result, Quantity
from .timeseries import TimeSeries
from .types import GeometryType, Period
from .model.grid import GridModelResult
from .model.dfsu import DfsuModelResult
from .model.track import TrackModelResult
from .model.point import PointModelResult
from .model.dummy import DummyModelResult
from .obs import Observation, PointObservation, TrackObservation
from .comparison import Comparer, ComparerCollection
from . import __version__

TimeDeltaTypes = Union[float, int, np.timedelta64, pd.Timedelta, timedelta]
IdxOrNameTypes = Optional[Union[int, str]]
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
    DummyModelResult,
]
ObsInputType = Union[
    str,
    Path,
    mikeio.DataArray,
    mikeio.Dataset,
    mikeio.Dfs0,
    pd.DataFrame,
    pd.Series,
    Observation,
]

T = TypeVar("T", bound="TimeSeries")


def from_matched(
    data: Union[str, Path, pd.DataFrame, mikeio.Dfs0, mikeio.Dataset],
    *,
    obs_item: str | int | None = 0,
    mod_items: Optional[Iterable[str | int]] = None,
    aux_items: Optional[Iterable[str | int]] = None,
    quantity: Optional[Quantity] = None,
    name: Optional[str] = None,
    weight: float = 1.0,
    x: Optional[float] = None,
    y: Optional[float] = None,
    z: Optional[float] = None,
    x_item: str | int | None = None,
    y_item: str | int | None = None,
) -> Comparer:
    """Create a Comparer from observation and model results that are already matched (aligned)
    Parameters
    ----------
    data : [pd.DataFrame, str, Path, mikeio.Dfs0, mikeio.Dataset]
        DataFrame (or object that can be converted to a DataFrame e.g. dfs0)
        with columns obs_item, mod_items, aux_items
    obs_item : [str, int], optional
        Name or index of observation item, by default first item
    mod_items : Iterable[str, int], optional
        Names or indicies of model items, if None all remaining columns are model items, by default None
    aux_items : Iterable[str, int], optional
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
    x_item: [str, int], optional,
        Name of x item, only relevant for track data
    y_item: [str, int], optional
        Name of y item, only relevant for track data

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
            quantity = Quantity.from_mikeio_iteminfo(data[obs_item].item)
        data = data.to_dataframe()

    cmp = Comparer.from_matched_data(
        data,
        obs_item=obs_item,
        mod_items=mod_items,
        aux_items=aux_items,
        name=name,
        weight=weight,
        x=x,
        y=y,
        z=z,
        x_item=x_item,
        y_item=y_item,
        quantity=quantity,
    )

    return cmp


@overload
def match(
    obs: PointObservation | TrackObservation,
    mod: Union[MRInputType, Sequence[MRInputType]],
    *,
    obs_item: Optional[IdxOrNameTypes] = None,
    mod_item: Optional[IdxOrNameTypes] = None,
    gtype: Optional[GeometryTypes] = None,
    max_model_gap: Optional[float] = None,
    spatial_method: Optional[str] = None,
) -> Comparer: ...


@overload
def match(
    obs: Iterable[PointObservation | TrackObservation],
    mod: Union[MRInputType, Sequence[MRInputType]],
    *,
    obs_item: Optional[IdxOrNameTypes] = None,
    mod_item: Optional[IdxOrNameTypes] = None,
    gtype: Optional[GeometryTypes] = None,
    max_model_gap: Optional[float] = None,
    spatial_method: Optional[str] = None,
) -> ComparerCollection: ...


def match(
    obs,
    mod,
    *,
    obs_item=None,
    mod_item=None,
    gtype=None,
    max_model_gap=None,
    spatial_method: Optional[str] = None,
):
    """Match observation and model result data in space and time

    NOTE: In case of multiple model results with different time coverage,
    only the _overlapping_ time period will be used! (intersection)

    NOTE: In case of multiple observations, multiple models can _only_
    be matched if they are _all_ of SpatialField type, e.g. DfsuModelResult
    or GridModelResult.

    Parameters
    ----------
    obs : (str, Path, pd.DataFrame, Observation, Sequence[Observation])
        Observation(s) to be compared
    mod : (str, Path, pd.DataFrame, ModelResult, Sequence[ModelResult])
        Model result(s) to be compared
    obs_item : int or str, optional
        observation item if obs is a file/dataframe, by default None
    mod_item : (int, str), optional
        model item if mod is a file/dataframe, by default None
    gtype : (str, optional)
        Geometry type of the model result (if mod is a file/dataframe).
        If not specified, it will be guessed.
    max_model_gap : (float, optional)
        Maximum time gap (s) in the model result (e.g. for event-based
        model results), by default None
    spatial_method : str, optional
        For Dfsu- and GridModelResult, spatial interpolation/selection method.

        - For DfsuModelResult, one of: 'contained' (=isel), 'nearest',
        'inverse_distance' (with 5 nearest points), by default "inverse_distance".
        - For GridModelResult, passed to xarray.interp() as method argument,
        by default 'linear'.

    Returns
    -------
    Comparer
        In case of a single observation
    ComparerCollection
        In case of multiple observations

    See Also
    --------
    [from_matched][modelskill.from_matched]
        Create a Comparer from observation and model results that are already matched
    """
    if isinstance(obs, get_args(ObsInputType)):
        return _single_obs_compare(
            obs,
            mod,
            obs_item=obs_item,
            mod_item=mod_item,
            gtype=gtype,
            max_model_gap=max_model_gap,
            spatial_method=spatial_method,
        )

    assert isinstance(obs, Collection)

    if len(obs) > 1 and isinstance(mod, Collection) and len(mod) > 1:
        if not all(isinstance(m, (DfsuModelResult, GridModelResult)) for m in mod):
            raise ValueError(
                """
                In case of multiple observations, multiple models can _only_ 
                be matched if they are _all_ of SpatialField type, e.g. DfsuModelResult 
                or GridModelResult. 
                
                If you want match multiple point observations with multiple point model results, 
                please match one observation at a time and then create a collection of these 
                using modelskill.ComparerCollection(cmp_list) afterwards. The same applies to track data.
                """
            )

    clist = [
        _single_obs_compare(
            o,
            mod,
            obs_item=obs_item,
            mod_item=mod_item,
            gtype=gtype,
            max_model_gap=max_model_gap,
            spatial_method=spatial_method,
        )
        for o in obs
    ]

    return ComparerCollection(clist)


def compare(
    obs,
    mod,
    *,
    obs_item=None,
    mod_item=None,
    gtype=None,
    max_model_gap=None,
) -> ComparerCollection:
    warnings.warn("compare is deprecated. Use match instead.", FutureWarning)
    observations = [obs] if isinstance(obs, get_args(ObsInputType)) else obs
    assert isinstance(observations, Iterable)

    clist = [
        _single_obs_compare(
            o,
            mod,
            obs_item=obs_item,
            mod_item=mod_item,
            gtype=gtype,
            max_model_gap=max_model_gap,
        )
        for o in observations
    ]

    return ComparerCollection(clist)


def _single_obs_compare(
    obs: ObsInputType,
    mod: Union[MRInputType, Sequence[MRInputType]],
    *,
    obs_item: Optional[int | str] = None,
    mod_item: Optional[int | str] = None,
    gtype: Optional[GeometryTypes] = None,
    max_model_gap: Optional[float] = None,
    spatial_method: Optional[str] = None,
) -> Comparer:
    """Compare a single observation with multiple models"""
    obs = _parse_single_obs(obs, obs_item, gtype=gtype)

    mods = _parse_models(mod, mod_item, gtype=gtype)

    raw_mod_data = {m.name: m.extract(obs, spatial_method) for m in mods}
    matched_data = match_space_time(
        observation=obs, raw_mod_data=raw_mod_data, max_model_gap=max_model_gap
    )
    matched_data.attrs["weight"] = obs.weight

    return Comparer(matched_data=matched_data, raw_mod_data=raw_mod_data)


def _get_global_start_end(idxs: Iterable[pd.DatetimeIndex]) -> Period:
    assert all([len(x) > 0 for x in idxs])

    starts = [x[0] for x in idxs]
    ends = [x[-1] for x in idxs]

    return Period(start=min(starts), end=max(ends))


def match_space_time(
    observation: PointObservation | TrackObservation,
    raw_mod_data: Dict[str, PointModelResult | TrackModelResult],
    max_model_gap: float | None = None,
    spatial_tolerance: float = 1e-3,
) -> xr.Dataset:
    """Match observation with one or more model results in time domain
    and return as xr.Dataset in the format used by modelskill.Comparer

    Will interpolate model results to observation time.

    Note: assumes that observation and model data are already matched in space.
        But positions of track observations will be checked.

    Parameters
    ----------
    observation : Observation
        Observation to be matched
    raw_mod_data : Dict[str, PointModelResult | TrackModelResult]
        Dictionary of model results ready for interpolation
    max_model_gap : Optional[TimeDeltaTypes], optional
        In case of non-equidistant model results (e.g. event data),
        max_model_gap can be given e.g. as seconds, by default None
    spatial_tolerance : float, optional
        Tolerance for spatial matching, by default 1e-3

    Returns
    -------
    xr.Dataset
        Matched data in the format used by modelskill.Comparer
    """
    obs_name = "Observation"
    mod_names = list(raw_mod_data.keys())
    idxs = [m.time for m in raw_mod_data.values()]
    period = _get_global_start_end(idxs)

    assert isinstance(observation, (PointObservation, TrackObservation))
    gtype = "point" if isinstance(observation, PointObservation) else "track"
    observation = observation.trim(period.start, period.end)

    data = observation.data
    data.attrs["name"] = observation.name
    data = data.rename({observation.name: obs_name})

    for _, mr in raw_mod_data.items():
        if isinstance(mr, PointModelResult):
            assert len(observation.time) > 0
            mri: TimeSeries = mr.interp_time(
                new_time=observation.time, max_gap=max_model_gap
            )
        else:
            mri = mr

        if isinstance(observation, TrackObservation):
            assert isinstance(mri, TrackModelResult)
            mri.data = _select_overlapping_trackdata_with_tolerance(
                observation=observation, mri=mri, spatial_tolerance=spatial_tolerance
            )

        # check that model and observation have non-overlapping variables
        if overlapping_names := set(mri.data.data_vars).intersection(
            set(data.data_vars)
        ):
            raise ValueError(
                f"Model: '{mr.name}' and observation have overlapping variables: {overlapping_names}"
            )

        # TODO: is name needed?
        for v in list(mri.data.data_vars):
            data[v] = mri.data[v]

    # drop NaNs in model and observation columns (but allow NaNs in aux columns)
    cols = list(
        data.filter_by_attrs(kind=lambda k: k in ["model", "observation"]).data_vars
    )
    data = data.dropna(dim="time", subset=cols)

    for n in mod_names:
        data[n].attrs["kind"] = "model"

    data.attrs["gtype"] = gtype
    data.attrs["modelskill_version"] = __version__

    return data


# TODO move to TrackModelResult
def _select_overlapping_trackdata_with_tolerance(
    observation: TrackObservation, mri: TrackModelResult, spatial_tolerance: float
) -> xr.Dataset:
    mod_df = mri.data.to_dataframe()
    obs_df = observation.data.to_dataframe()

    # 1. inner join on time
    df = mod_df.join(obs_df, how="inner", lsuffix="_mod", rsuffix="_obs")

    # 2. remove model points outside observation track
    n_points = len(df)
    keep_x = np.abs((df.x_mod - df.x_obs)) < spatial_tolerance
    keep_y = np.abs((df.y_mod - df.y_obs)) < spatial_tolerance
    df = df[keep_x & keep_y]
    if n_points_removed := n_points - len(df):
        warnings.warn(
            f"Removed {n_points_removed} model points outside observation track (spatial_tolerance={spatial_tolerance})"
        )
    return mri.data.sel(time=df.index)


def _parse_single_obs(
    obs: ObsInputType,
    item: Optional[int | str] = None,
    gtype: Optional[GeometryTypes] = None,
) -> PointObservation | TrackObservation:
    if isinstance(obs, (PointObservation, TrackObservation)):
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
    mod: Any,  # TODO
    item: Optional[IdxOrNameTypes] = None,
    gtype: Optional[GeometryTypes] = None,
) -> List[Any]:  # TODO
    """Return a list of ModelResult objects"""
    if isinstance(mod, get_args(MRInputType)):
        return [_parse_single_model(mod, item=item, gtype=gtype)]
    elif isinstance(mod, Sequence):
        return [_parse_single_model(m, item=item, gtype=gtype) for m in mod]
    else:
        raise ValueError(f"Unknown mod type {type(mod)}")


def _parse_single_model(
    mod: Any,  # TODO
    item: Optional[IdxOrNameTypes] = None,
    gtype: Optional[GeometryTypes] = None,
) -> Any:  # TODO
    if isinstance(
        mod,
        (
            str,
            Path,
            pd.DataFrame,
            xr.Dataset,
            xr.DataArray,
            mikeio.Dfs0,
            mikeio.Dataset,
            mikeio.DataArray,
            mikeio.dfsu.Dfsu2DH,
        ),
    ):
        try:
            return model_result(mod, item=item, gtype=gtype)
        except ValueError as e:
            raise ValueError(
                f"Could not compare. Unknown model result type {type(mod)}. {str(e)}"
            )
    else:
        if item is not None:
            raise ValueError("item argument not allowed if mod is a ModelResult type")
        # assume it is already a model result
        return mod
