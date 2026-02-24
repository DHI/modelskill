from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import (
    Collection,
    Iterable,
    Literal,
    Mapping,
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

from modelskill.model.point import PointModelResult

from . import Quantity
from .comparison import Comparer, ComparerCollection
from .model.dfsu import DfsuModelResult
from .model.dummy import DummyModelResult
from .model.grid import GridModelResult
from .model.network import NetworkModelResult, NodeModelResult
from .model.track import TrackModelResult
from .model.point import align_data
from .obs import (
    Observation,
    PointObservation,
    TrackObservation,
    NodeObservation,
)
from .timeseries import TimeSeries
from .types import Period

TimeDeltaTypes = Union[float, int, np.timedelta64, pd.Timedelta, timedelta]
IdxOrNameTypes = Union[int, str] | None
GeometryTypes = Literal["point", "track", "unstructured", "grid"] | None
MRTypes = Union[
    PointModelResult,
    GridModelResult,
    DfsuModelResult,
    TrackModelResult,
    NetworkModelResult,
    DummyModelResult,
]
FieldTypes = Union[
    GridModelResult,
    DfsuModelResult,
]
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
    MRTypes,
]
ObsTypes = Union[PointObservation, TrackObservation, NodeObservation]
ObsInputType = Union[
    str,
    Path,
    mikeio.DataArray,
    mikeio.Dataset,
    mikeio.Dfs0,
    pd.DataFrame,
    pd.Series,
    ObsTypes,
]
T = TypeVar("T", bound="TimeSeries")


def from_matched(
    data: Union[str, Path, pd.DataFrame, mikeio.Dfs0, mikeio.Dataset],
    *,
    obs_item: str | int | None = 0,
    mod_items: Iterable[str | int] | None = None,
    aux_items: Iterable[str | int] | None = None,
    quantity: Quantity | None = None,
    name: str | None = None,
    weight: float = 1.0,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
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

    Returns
    -------
    Comparer
        A Comparer object with matched observation and model data

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
    obs: ObsTypes,
    mod: MRTypes | Sequence[MRTypes],
    *,
    max_model_gap: float | None = None,
    spatial_method: str | None = None,
    spatial_tolerance: float = 1e-3,
    obs_no_overlap: Literal["ignore", "error", "warn"] = "error",
) -> Comparer: ...


@overload
def match(
    obs: Iterable[ObsTypes],
    mod: MRTypes | Sequence[MRTypes],
    *,
    max_model_gap: float | None = None,
    spatial_method: str | None = None,
    spatial_tolerance: float = 1e-3,
    obs_no_overlap: Literal["ignore", "error", "warn"] = "error",
) -> ComparerCollection: ...


def match(
    obs,
    mod,
    *,
    max_model_gap=None,
    spatial_method: str | None = None,
    spatial_tolerance: float = 1e-3,
    obs_no_overlap: Literal["ignore", "error", "warn"] = "error",
):
    """Match observation and model result data in space and time

    NOTE: In case of multiple model results with different time coverage,
    only the _overlapping_ time period will be used! (intersection)

    NOTE: In case of multiple observations, multiple models can _only_
    be matched if they are _all_ of SpatialField type, e.g. DfsuModelResult
    or GridModelResult.

    Parameters
    ----------
    obs : (Observation, Sequence[Observation])
        Observation(s) to be compared
    mod : (ModelResult, Sequence[ModelResult])
        Model result(s) to be compared
    max_model_gap : (float, optional)
        Maximum time gap (s) in the model result (e.g. for event-based
        model results), by default None
    spatial_method : str, optional
        For Dfsu- and GridModelResult, spatial interpolation/selection method.

        - For DfsuModelResult, one of: 'contained' (=isel), 'nearest',
        'inverse_distance' (with 5 nearest points), by default "inverse_distance".
        - For GridModelResult, passed to xarray.interp() as method argument,
        by default 'linear'.
    spatial_tolerance : float, optional
        Spatial tolerance (in the units of the coordinate system) for matching
        model track points to observation track points. Model points outside
        this tolerance will be discarded. Only relevant for TrackModelResult
        and TrackObservation, by default 1e-3.
    obs_no_overlap: str, optional
        How to handle observations with no overlap with model results. One of: 'ignore', 'error', 'warn', by default 'error'.

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
        return _match_single_obs(
            obs,
            mod,
            max_model_gap=max_model_gap,
            spatial_method=spatial_method,
            spatial_tolerance=spatial_tolerance,
            obs_no_overlap=obs_no_overlap,
        )

    if isinstance(obs, Collection):
        assert all(isinstance(o, get_args(ObsInputType)) for o in obs)
    else:
        raise TypeError(
            f"Obs is not the correct type: it is {type(obs)}. Check the order of the arguments (obs, mod)."
        )

    if len(obs) > 1 and isinstance(mod, Collection) and len(mod) > 1:
        if not all(
            isinstance(
                m,
                (
                    DfsuModelResult,
                    GridModelResult,
                    NetworkModelResult,
                    DummyModelResult,
                ),
            )
            for m in mod
        ):
            raise ValueError(
                """
                When matching multiple observations with multiple models, all models
                must be one of the following types: DfsuModelResult, GridModelResult or NetworkModelResult.
                
                If you want match multiple point observations with multiple point model results, 
                please match one observation at a time and then create a collection of these 
                using modelskill.ComparerCollection(cmp_list) afterwards. The same applies to track data.
                """
            )

    clist = [
        _match_single_obs(
            o,
            mod,
            max_model_gap=max_model_gap,
            spatial_method=spatial_method,
            spatial_tolerance=spatial_tolerance,
            obs_no_overlap=obs_no_overlap,
        )
        for o in obs
    ]

    cmps = [c for c in clist if c is not None]

    return ComparerCollection(cmps)


def _match_single_obs(
    obs: ObsTypes,
    mod: MRTypes | Sequence[MRTypes],
    *,
    max_model_gap: float | None,
    spatial_method: str | None,
    spatial_tolerance: float,
    obs_no_overlap: Literal["ignore", "error", "warn"],
) -> Comparer | None:
    if isinstance(mod, get_args(MRInputType)):
        models: list = [mod]
    else:
        models = mod  # type: ignore

    names = [m.name for m in models]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate model names found: {names}")

    raw_mod_data: dict[str, PointModelResult | TrackModelResult | NodeModelResult] = {}
    for m in models:
        is_field = isinstance(m, (GridModelResult, DfsuModelResult))
        is_dummy = isinstance(m, DummyModelResult)
        is_network = isinstance(m, NetworkModelResult)
        if is_field or is_dummy:
            matching_obs = m.extract(obs, spatial_method=spatial_method)
        elif is_network:
            matching_obs = m.extract(obs)
        else:
            matching_obs = m

        raw_mod_data[m.name] = matching_obs

    matched_data = _match_space_time(
        observation=obs,
        raw_mod_data=raw_mod_data,
        max_model_gap=max_model_gap,
        obs_no_overlap=obs_no_overlap,
        spatial_tolerance=spatial_tolerance,
    )
    if matched_data is None:
        return None
    matched_data.attrs["weight"] = obs.weight

    return Comparer(matched_data=matched_data, raw_mod_data=raw_mod_data)


def _get_global_start_end(idxs: Iterable[pd.DatetimeIndex]) -> Period:
    assert all([len(x) > 0 for x in idxs]), "All datetime indices must be non-empty"

    starts = [x[0] for x in idxs]
    ends = [x[-1] for x in idxs]

    return Period(start=min(starts), end=max(ends))


def _match_space_time(
    observation: Observation,
    raw_mod_data: Mapping[
        str, PointModelResult | TrackModelResult | NetworkModelResult | NodeModelResult
    ],
    max_model_gap: float | None,
    spatial_tolerance: float,
    obs_no_overlap: Literal["ignore", "error", "warn"],
) -> xr.Dataset | None:
    idxs = [m.time for m in raw_mod_data.values()]
    period = _get_global_start_end(idxs)

    observation = observation.trim(period.start, period.end, no_overlap=obs_no_overlap)
    if len(observation.data.time) == 0:
        return None

    data = observation.data.copy()
    data.attrs["name"] = observation.name
    data = data.rename({observation.name: "Observation"})

    for mr in raw_mod_data.values():
        match mr, observation:
            case TrackModelResult() as tmr, TrackObservation():
                aligned = tmr.subset_to(
                    observation, spatial_tolerance=spatial_tolerance
                )
            case PointModelResult() as pmr, PointObservation():
                aligned = align_data(pmr.data, observation, max_gap=max_model_gap)
            case NodeModelResult() as nmr, NodeObservation():
                # mr is the extracted NodeModelResult
                aligned = align_data(nmr.data, observation, max_gap=max_model_gap)
            case _:
                raise TypeError(
                    f"Matching not implemented for model type {type(mr)} and observation type {type(observation)}"
                )

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
