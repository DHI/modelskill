from __future__ import annotations
from pathlib import Path

from typing import (
    Iterable,
    List,
    Literal,
    Optional,
    Union,
    Sequence,
    get_args,
)
import warnings
import pandas as pd
import xarray as xr

import mikeio

from modelskill import ModelResult
from modelskill.timeseries import TimeSeries
from modelskill.types import GeometryType, Quantity
from .model import protocols
from .model.grid import GridModelResult
from .model.dfsu import DfsuModelResult
from .model.track import TrackModelResult
from .observation import Observation, PointObservation, TrackObservation
from .comparison import Comparer, ComparerCollection


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
) -> Union[Comparer, ComparerCollection]:
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
    Comparer or ComparerCollection
        To be used for plotting and statistics
    """
    if isinstance(obs, get_args(ObsInputType)):
        return _single_obs_compare(
            obs,
            mod,
            obs_item=obs_item,
            mod_item=mod_item,
            gtype=gtype,
            max_model_gap=max_model_gap,
        )
    elif isinstance(obs, Sequence):
        clist = [
            compare(
                o,
                mod,
                obs_item=obs_item,
                mod_item=mod_item,
                gtype=gtype,
                max_model_gap=max_model_gap,
            )
            for o in obs
        ]
        return ComparerCollection(clist)
    else:
        raise ValueError(f"Unknown obs type {type(obs)}")


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
    df_mod = _extract_from_models(obs, mod)  # type: ignore

    return Comparer(obs, df_mod, max_model_gap=max_model_gap)


def _parse_single_obs(
    obs, item=None, gtype: Optional[GeometryTypes] = None
) -> protocols.Observation:
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


def _extract_from_models(obs, mod: List[protocols.ModelResult]) -> List[pd.DataFrame]:
    df_model = []
    for mr in mod:
        if hasattr(mr, "extract"):
            mr = mr.extract(obs)

        df = mr.data

        # TODO is this robust enough?
        old_item = df.columns.values[-1]
        df = df.rename(columns={old_item: mr.name})
        if (df is not None) and (len(df) > 0):
            df_model.append(df)
        else:
            warnings.warn(
                f"No data found when extracting '{obs.name}' from model '{mr.name}'"
            )
    return df_model
