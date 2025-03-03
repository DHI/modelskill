from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import (
    Any,
    Collection,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
    get_args,
    overload,
)

import mikeio
import numpy as np
import pandas as pd
import xarray as xr

from . import Quantity, __version__, model_result
from .comparison import Comparer, ComparerCollection
from .model._base import Alignable
from .model.dfsu import DfsuModelResult
from .model.dummy import DummyModelResult
from .model.grid import GridModelResult
from .model.track import TrackModelResult
from .obs import Observation, observation
from .timeseries import TimeSeries
from .types import Period

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
    """Create a Comparer from data that is already matched (aligned).

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
        if Path(data).suffix != ".dfs0":
            raise ValueError(f"File must be a dfs0 file, not {Path(data).suffix}")
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
    obs: Observation,
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
    obs: Iterable[Observation],
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
    from_matched - Create a Comparer from observation and model results that are already matched
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

    if isinstance(obs, Collection):
        assert all(isinstance(o, get_args(ObsInputType)) for o in obs)
    else:
        raise TypeError(
            f"Obs is not the correct type: it is {type(obs)}. Check the order of the arguments (obs, mod)."
        )

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

    # TODO where does this line belong?
    matched_data.attrs["modelskill_version"] = __version__

    return Comparer(matched_data=matched_data, raw_mod_data=raw_mod_data)


def _get_global_start_end(idxs: Iterable[pd.DatetimeIndex]) -> Period:
    assert all([len(x) > 0 for x in idxs])

    starts = [x[0] for x in idxs]
    ends = [x[-1] for x in idxs]

    return Period(start=min(starts), end=max(ends))


def match_space_time(
    observation: Observation,
    raw_mod_data: Mapping[str, Alignable],
    max_model_gap: float | None = None,
) -> xr.Dataset:
    """Match observation with one or more model results in time domain.

    and return as xr.Dataset in the format used by modelskill.Comparer

    Will interpolate model results to observation time.

    Note: assumes that observation and model data are already matched in space.
        But positions of track observations will be checked.

    Parameters
    ----------
    observation : Observation
        Observation to be matched
    raw_mod_data : Mapping[str, Alignable]
        Mapping of model results ready for interpolation
    max_model_gap : Optional[TimeDeltaTypes], optional
        In case of non-equidistant model results (e.g. event data),
        max_model_gap can be given e.g. as seconds, by default None

    Returns
    -------
    xr.Dataset
        Matched data in the format used by modelskill.Comparer
    """
    idxs = [m.time for m in raw_mod_data.values()]
    period = _get_global_start_end(idxs)

    observation = observation.trim(period.start, period.end)

    data = observation.data.copy()
    data.attrs["name"] = observation.name
    data = data.rename({observation.name: "Observation"})

    for mr in raw_mod_data.values():
        # TODO is `align` the correct name for this operation?
        aligned = mr.align(observation, max_gap=max_model_gap)

        if overlapping := set(aligned.filter_by_attrs(kind="aux").data_vars) & set(
            observation.data.filter_by_attrs(kind="aux").data_vars
        ):
            raise ValueError(
                f"Aux variables are not allowed to have identical names. Choose either aux from obs or model. Overlapping: {overlapping}"
            )

        for dv in aligned:
            data[dv] = aligned[dv]

    # drop NaNs in model and observation columns (but allow NaNs in aux columns)
    def mo_kind(k: str) -> bool:
        return k in ["model", "observation"]

    # TODO mo_cols vs non_aux_cols?
    mo_cols = data.filter_by_attrs(kind=mo_kind).data_vars
    data = data.dropna(dim="time", subset=mo_cols)

    return data


def _parse_single_obs(
    obs: ObsInputType,
    obs_item: Optional[int | str],
    gtype: Optional[GeometryTypes],
) -> Observation:
    if isinstance(obs, Observation):
        if obs_item is not None:
            raise ValueError(
                "obs_item argument not allowed if obs is an modelskill.Observation type"
            )
        return obs
    else:
        # observation factory can only handle track and point
        return observation(obs, item=obs_item, gtype=gtype)  # type: ignore


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
