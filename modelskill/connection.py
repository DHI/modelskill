from abc import ABC, abstractmethod
import os
from pathlib import Path

import yaml
from typing import Iterable, List, Literal, Optional, Union, Mapping, Sequence, get_args
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import mikeio

from modelskill import ModelResult
from modelskill.timeseries import TimeSeries
from modelskill.model import PointModelResult
from modelskill.types import GeometryType, Quantity
from .model import protocols
from .model._base import ModelResultBase
from .observation import Observation, PointObservation, TrackObservation
from .comparison import Comparer, PointComparer, ComparerCollection, TrackComparer
from .utils import is_iterable_not_str
from .plot import plot_spatial_overview


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
    ModelResultBase,
    TimeSeries
    # protocols.ModelResult,
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
    df: pd.DataFrame,
    *,
    obs_item: str,
    mod_items: Optional[Iterable[str]] = None,
    quantity: Optional[Quantity] = None,
) -> Comparer:
    """Create a Comparer from observation and model results that are already matched (aligned)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns [obs_item, mod_item]
    obs_item : str
        Name of observation item
    mod_items : Optional[Iterable[str]], optional
        Names of model items, if None all remaining columns are model items, by default None
    quantity : Quantity, optional
        Quantity of the observation and model results, by default Quantity(name="Undefined", unit="Undefined")

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
    obs = df[obs_item]

    if mod_items is None:
        # all remaining columns are model results
        pmods = [
            PointModelResult(df[c], item=c, quantity=quantity)
            for c in df.columns
            if c != obs_item
        ]
    else:
        pmods = [PointModelResult(df[c], item=c, quantity=quantity) for c in mod_items]
    pobs = PointObservation(obs, item=obs_item, name=obs_item, quantity=quantity)

    return _single_obs_compare(pobs, pmods)


def compare(
    obs: Union[ObsInputType, Sequence[ObsInputType]],
    mod: Union[MRInputType, Sequence[MRInputType]],
    *,
    obs_item: IdOrNameTypes = None,
    mod_item: IdOrNameTypes = None,
    gtype: GeometryTypes = None,
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
    gtype: GeometryTypes = None,
    max_model_gap=None,
) -> Comparer:
    """Compare a single observation with multiple models"""
    obs = _parse_single_obs(obs, obs_item, gtype=gtype)
    mod = _parse_models(mod, mod_item, gtype=gtype)
    df_mod = _extract_from_models(obs, mod)

    return Comparer(obs, df_mod, max_model_gap=max_model_gap)


def _parse_single_obs(
    obs, item: IdOrNameTypes = None, gtype: GeometryTypes = None
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
    mod, item: IdOrNameTypes = None, gtype: GeometryTypes = None
) -> List[protocols.ModelResult]:
    """Return a list of ModelResult objects"""
    if isinstance(mod, get_args(MRInputType)):
        return [_parse_single_model(mod, item=item, gtype=gtype)]
    elif isinstance(mod, Sequence):
        return [_parse_single_model(m, item=item, gtype=gtype) for m in mod]
    else:
        raise ValueError(f"Unknown mod type {type(mod)}")


def _parse_single_model(
    mod, item: IdOrNameTypes = None, gtype: GeometryTypes = None
) -> protocols.ModelResult:
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
        if (df is not None) and (len(df) > 0):
            df_model.append(df)
        else:
            warnings.warn(
                f"No data found when extracting '{obs.name}' from model '{mr.name}'"
            )
    return df_model


# def _parse_model(mod, item: IdOrNameTypes = None) -> protocols.ModelResult:
#     if isinstance(mod, str):
#         dfs = mikeio.open(mod)
#         if (len(dfs.items) > 1) and (item is None):
#             raise ValueError("Model ambiguous - please provide item")
#         mod = dfs.read(items=item).to_dataframe()
#     elif isinstance(mod, pd.DataFrame):
#         mod = ModelResult(mod, item=item).data
#     elif isinstance(mod, pd.Series):
#         mod = mod.to_frame()

#     assert mod.shape[1] == 1  # A single item

#     mod.columns = ["Model"]

#     return mod


class _BaseConnector(ABC):
    def __init__(self) -> None:
        self.modelresults = {}
        self.name = None
        self.obs = None

    @property
    def n_models(self):
        """Number of (unique) model results in Connector."""
        return len(self.modelresults)

    @property
    def mod_names(self):
        """Names of (unique) model results in Connector."""
        return list(self.modelresults.keys())

    @abstractmethod
    def extract(self):
        pass


class _SingleObsConnector(_BaseConnector):
    """A connection between a single observation and model(s)"""

    def __repr__(self):
        if self.n_models > 0:
            obs_txt = f"obs={self.name}(n={self.obs.n_points})"
            mod_txt = f"model={self.modelresults[0].name}"
            if self.n_models > 1:
                mod_txt = f"{self.n_models} models="
                mod_txt += "[" + ", ".join(m.name for m in self.modelresults) + "]"
                if len(mod_txt) > 25:
                    mod_txt = mod_txt[:20] + "...]"
            txt = f"{obs_txt} :: {mod_txt}"
        else:
            txt = "[empty]"

        return f"<{self.__class__.__name__}> {txt}"

    def __init__(self, obs, mod, weight=1.0, validate=True):
        super().__init__()
        obs = self._parse_observation(obs)
        self.name = obs.name
        modelresults = self._parse_model(mod)

        ok = self._validate(obs, modelresults)
        if validate and (not ok):
            mod_txt = (
                f"model '{modelresults[0].name}'"
                if len(modelresults) == 1
                else f"models {[m.name for m in modelresults]}"
            )
            raise ValueError(
                f"Validation failed! Cannot connect observation '{obs.name}' and {mod_txt}."
            )
        if ok or (not validate):
            self.modelresults = modelresults
            self.obs = obs
            self.obs.weight = weight

    def _parse_model(self, mod) -> List[protocols.ModelResult]:
        if is_iterable_not_str(mod):
            mr = []
            for m in mod:
                mr.append(self._parse_single_model(m))
        else:
            mr = [self._parse_single_model(mod)]
        return mr

    def _parse_single_model(self, mod) -> protocols.ModelResult:
        if isinstance(mod, protocols.ModelResult):
            return mod
        elif isinstance(mod, (pd.Series, pd.DataFrame, mikeio.DataArray)):
            return ModelResult(mod)
        else:
            raise ValueError(f"Unknown model result type {type(mod)}")

    def _validate(self, obs, modelresults):
        # TODO: add validation errors to list
        ok = True
        for mod in modelresults:
            # has_mod_item = self._has_mod_item(mod)
            quantity_match = obs.quantity.is_compatible(mod.quantity)
            in_domain = self._validate_in_domain(obs, mod)
            time_overlaps = self._validate_start_end(obs, mod)
            ok = ok and quantity_match and in_domain and time_overlaps
        return ok

    @staticmethod
    def _validate_in_domain(obs, mod):
        in_domain = True
        # if isinstance(mod, protocols.ModelResult) and isinstance(obs, PointObservation):
        #     in_domain = mod._in_domain(obs.x, obs.y)
        #     if not in_domain:
        #         warnings.warn(
        #             f"Outside domain! Obs '{obs.name}' outside model '{mod.name}'"
        #         )
        return in_domain

    @staticmethod
    def _validate_start_end(obs, mod):

        if obs.end_time < mod.start_time:
            warnings.warn(
                f"No time overlap! Obs '{obs.name}' end is before model '{mod.name}' start"
            )
            return False

        if obs.start_time > mod.end_time:
            warnings.warn(
                f"No time overlap! Obs '{obs.name}' start is after model '{mod.name}' end"
            )
            return False

        return True

    def plot_observation_positions(self, figsize=None):
        """Plot observation points on a map showing the model domain

        Parameters
        ----------
        figsize : (float, float), optional
            figure size, by default None
        """
        return plot_spatial_overview(
            obs=[self.obs], mod=self.modelresults, figsize=figsize
        )

    @staticmethod
    def _comparer_or_None(comparer, warn=True):
        """If comparer is empty issue warning and return None."""
        if comparer.n_points == 0:
            if warn:
                name = comparer.name
                warnings.warn(f"No overlapping data was found for {name}!")
            return None
        return comparer


class PointConnector(_SingleObsConnector):
    """Connector for a single PointObservation and ModelResults

    Typically, not constructed directly, but part of a Connector.

    Examples
    --------
    >>> mr = ModelResult("Oresund2D.dfsu", item=0)
    >>> o1 = PointObservation("Drogden_Fyr.dfs0", item=0, x=355568., y=6156863.)
    >>> con1 = PointConnector(o1, mr)
    >>> con = Connector(o1, mr)    # con[0] = con1
    """

    def _parse_observation(self, obs) -> PointObservation:
        if isinstance(obs, (pd.Series, pd.DataFrame)):
            return PointObservation(obs)
        elif isinstance(obs, str):
            return PointObservation(obs)
        elif isinstance(obs, Observation):
            return obs
        else:
            raise ValueError(f"Unknown observation type {type(obs)}")

    def extract(self, max_model_gap: float = None) -> PointComparer:
        """Extract model results at times and positions of observation.

        Returns
        -------
        PointComparer
            A comparer object for further analysis and plotting.
        """

        assert isinstance(self.obs, PointObservation)
        df_model = []
        for mr in self.modelresults:
            if hasattr(mr, "extract"):
                mr = mr.extract(self.obs)

            df = mr.data
            if (df is not None) and (len(df) > 0):
                df_model.append(df)
            else:
                warnings.warn(
                    f"No data found when extracting '{self.obs.name}' from model '{mr.name}'"
                )

        if len(df_model) == 0:
            warnings.warn(
                f"No overlapping data was found for PointObservation '{self.obs.name}'!"
            )
            return None

        comparer = PointComparer(self.obs, df_model, max_model_gap=max_model_gap)
        return self._comparer_or_None(comparer)


class TrackConnector(_SingleObsConnector):
    """Connector for a single TrackObservation and ModelResults

    Typically, not constructed directly, but part of a Connector.

    Examples
    --------
    >>> mr = ModelResult("Oresund2D.dfsu", item=0)
    >>> o1 = TrackObservation(df, item=2, name="altimeter")
    >>> con1 = TrackConnector(o1, mr)
    >>> con = Connector(o1, mr)    # con[0] = con1
    """

    def _parse_observation(self, obs) -> TrackObservation:
        if isinstance(obs, TrackObservation):
            return obs
        else:
            raise ValueError(f"Unknown track observation type {type(obs)}")

    def extract(self, max_model_gap: float = None) -> TrackComparer:
        """Extract model results at times and positions of track observation.

        Returns
        -------
        TrackComparer
            A comparer object for further analysis and plotting."""

        assert isinstance(self.obs, TrackObservation)
        df_model = []
        for mr in self.modelresults:
            if hasattr(mr, "extract"):
                mr = mr.extract(self.obs)

            df = mr.data
            if (df is not None) and (len(df) > 0):
                df_model.append(df)
            else:
                warnings.warn(
                    f"No data in extracted track '{self.obs.name}' from model '{mr.name}'"
                )

        if len(df_model) == 0:
            warnings.warn(
                f"No overlapping data was found for TrackObservation '{self.obs.name}'!"
            )
            # TODO returning None is not consistent with type hint
            return None

        comparer = TrackComparer(self.obs, df_model, max_model_gap=max_model_gap)
        return self._comparer_or_None(comparer)


class Connector(_BaseConnector, Mapping, Sequence):
    """The Connector is used for matching Observations and ModelResults

    It is one of the most important classes in modelskill. The connections are
    added either at construction of the Connector or by using the add()
    method.

    When observations and modelResults are added the connection is
    validated (inside domain? overlapping time? eum match?).

    The extract() method are then called to extract ModelResult data at
    the time and positions of each observation.

    Note
    ----
    Only ModelResults with a single item can be added to the Connector.
    From a multi-item ModelResult 'mr' an item must selected e.g. with
    'mr[0]' before adding

    Examples
    --------
    >>> mr = ModelResult("Oresund2D.dfsu", item=0)
    >>> o1 = PointObservation("Drogden_Fyr.dfs0", item=0, x=355568., y=6156863.)
    >>> o2 = TrackObservation(df, item=2, name="altimeter")
    >>> conA = Connector([o1, o2], mr)
    >>> conB = Connector()
    >>> conB.add(o1, mr)
    >>> conB.add(o2, mr)    # conA = conB
    >>> cc = conB.extract()
    """

    @property
    def n_observations(self):
        """Number of (unique) observations in Connector."""
        return len(self.observations)

    @property
    def obs_names(self):
        """Names of (unique) observations in Connector."""
        return list(self.observations.keys())

    def __repr__(self):
        txt = "<Connector> with \n"
        return txt + "\n".join(" -" + repr(c) for c in self.connections.values())

    def __init__(self, obs=None, mod=None, weight=1.0, validate=True):
        super().__init__()
        self.connections = {}
        self.observations = {}
        self.modelresults = {}
        if (obs is not None) and (mod is not None):
            if not is_iterable_not_str(obs):
                obs = [obs]
            weight = self._parse_weights(len(obs), weight)
            for j, o in enumerate(obs):
                self.add(o, mod, weight=weight[j], validate=validate)
        elif (mod is not None) or (obs is not None):
            raise ValueError("obs and mod must both be specified (or both None)")

    def add(self, obs, mod=None, weight=1.0, validate=True):
        """Add Observation-ModelResult-connections to Connector

        Note
        ----
        Only ModelResults with a single item can be added to the Connector.
        From a multi-item ModelResult 'mr' an item must selected e.g. with
        'mr[0]' before adding

        Parameters
        ----------
        obs : (str, pd.DataFrame, Observation)
            Observation(s) to be compared
        mod : (str, pd.DataFrame, ModelResult)
            Model result(s) to be compared
        weight: float, optional
            Relative weight used in weighted skill calculation, default 1.0
        validate : bool, optional
            Perform validation on eum type, observation-model
            overlap in space and time? by default True

        Examples
        --------
        >>> mr = ModelResult("Oresund2D.dfsu", item=0)
        >>> o1 = PointObservation("Drogden_Fyr.dfs0", item=0, x=355568., y=6156863.)
        >>> o2 = TrackObservation(df, item=2, name="altimeter")
        >>> conA = Connector()
        >>> conA.add([o1, o2], mr)
        >>> conB = Connector()
        >>> conB.add(o1, mr)
        >>> conB.add(o2, mr)   # conA = conB
        """
        if is_iterable_not_str(obs):
            weight = self._parse_weights(len(obs), weight)
            for j, o in enumerate(obs):
                self.add(o, mod, weight=weight[j], validate=validate)
            return
        elif isinstance(obs, _SingleObsConnector):
            con = obs
        else:
            if isinstance(obs, TrackObservation):
                con = TrackConnector(obs, mod, weight=weight, validate=validate)
            else:
                con = PointConnector(obs, mod, weight=weight, validate=validate)
        if con.n_models > 0:  # What other option is there??
            if con.name not in self.connections:
                self.connections[con.name] = con
                self._add_observation(con.obs)
            else:
                self.connections[con.name].modelresults.append(con.modelresults[0])

            self._add_modelresults(con.modelresults)

    @staticmethod
    def _parse_weights(n_obs, weights):
        if np.isscalar(weights):
            weights = weights * np.ones(n_obs, dtype=np.float64)
        if len(weights) != n_obs:
            raise ValueError("weight and obs should have same length")
        return weights

    def _add_observation(self, obs):
        if obs.name not in self.obs_names:
            self.observations[obs.name] = obs

    def _add_modelresults(self, mod):
        if is_iterable_not_str(mod):
            for m in mod:
                self._add_modelresults(m)
        else:
            if mod.name not in self.mod_names:
                self.modelresults[mod.name] = mod

    def _get_obs_name(self, obs):
        return self.obs_names[self._get_obs_id(obs)]

    def _get_obs_id(self, obs):
        n_con = len(self.connections)
        if obs is None or n_con <= 1:
            return 0
        elif isinstance(obs, str):
            if obs in self.obs_names:
                obs_id = self.obs_names.index(obs)
            else:
                raise KeyError(
                    f"connection {obs} could not be found in {self.obs_names}"
                )
        elif isinstance(obs, int):
            if obs < 0:  # Handle negative indices
                obs += n_con
            if obs >= 0 and obs < n_con:
                obs_id = obs
            else:
                raise IndexError(f"connection id {obs} is out of range (0, {n_con-1})")
        else:
            raise KeyError("connection must be None, str or int")
        return obs_id

    def __getitem__(self, x):
        if isinstance(x, int):
            x = self._get_obs_name(x)

        return self.connections[x]

    def __len__(self) -> int:
        return len(self.connections)

    def __iter__(self):
        return iter(self.connections.values())

    def extract(self, *args, **kwargs) -> ComparerCollection:
        """Extract model results at times and positions of all observations.

        Returns
        -------
        ComparerCollection
            A comparer object for further analysis and plotting.
        """

        cmps = [con.extract(*args, **kwargs) for con in self.connections.values()]
        cc = ComparerCollection(cmps)
        return cc

    def plot_observation_positions(self, title=None, figsize=None):
        """Plot observation points on a map showing the model domain

        Parameters
        ----------
        title: str, optional
            plot title, default empty
        figsize : (float, float), optional
            figure size, by default None

        Examples
        --------
        >>> con.plot_observation_positions()
        >>> con.plot_observation_positions(figsize=(10,10))
        >>> con.plot_observation_positions("A Map")
        """
        obs = list(self.observations.values())
        mod = list(self.modelresults.values())
        return plot_spatial_overview(obs=obs, mod=mod, title=title, figsize=figsize)

    def plot_temporal_coverage(
        self,
        *,
        show_model=True,
        limit_to_model_period=True,
        marker="_",
        title=None,
        figsize=None,
    ):
        """Plot graph showing temporal coverage for all observations

        Parameters
        ----------
        show_model : bool, optional
            Show model(s) as separate lines on plot, by default True
        limit_to_model_period : bool, optional
            Show temporal coverage only for period covered
            by the model, by default True
        marker : str, optional
            plot marker for observations, by default "_"
        title: str, optional
            plot title, default empty
        figsize : Tuple(float, float), optional
            size of figure, by default (7, 0.45*n_lines)

        Examples
        --------
        >>> con.plot_temporal_coverage()
        >>> con.plot_temporal_coverage(show_model=False)
        >>> con.plot_temporal_coverage(limit_to_model_period=False)
        >>> con.plot_temporal_coverage(marker=".")
        >>> con.plot_temporal_coverage(figsize=(5,3))
        """
        n_models = self.n_models if show_model else 0
        n_lines = n_models + self.n_observations
        if figsize is None:
            ysize = max(2.0, 0.45 * n_lines)
            figsize = (7, ysize)

        fig, ax = plt.subplots(figsize=figsize)
        y = np.repeat(0.0, 2)
        labels = []

        if show_model:
            for key, mr in self.modelresults.items():
                y += 1.0
                plt.plot([mr.start_time, mr.end_time], y)
                labels.append(key)

        for key, obs in self.observations.items():
            y += 1.0
            plt.plot(obs.time, y[0] * np.ones_like(obs.values), marker, markersize=5)
            labels.append(key)

        if limit_to_model_period:
            mr = list(self.modelresults.values())[0]  # take first
            plt.xlim([mr.start_time, mr.end_time])

        plt.yticks(np.arange(n_lines) + 1, labels)
        if show_model:
            for j in range(n_models):
                ax.get_yticklabels()[j].set_fontstyle("italic")
                ax.get_yticklabels()[j].set_weight("bold")
                # set_color("#004165")
        fig.autofmt_xdate()

        if title:
            ax.set_title(title)
        return ax

    def to_config(self, filename: str = None, relative_path=True):
        """Save Connector to a config file.

        Parameters
        ----------
        filename: str or Path
            Save configuration in yaml format
        relative_path: bool, default=True
            Use filenames relative to config file location

        Notes
        -----
        1. Manually create your Connector in modelskill as usual
        2. When you are satisfied, save config: connector.to_config('conf.yml')
        3. Later: run your reporting from the commandline e.g. directly after model execution
        """
        conf = {}

        folder = None
        if relative_path and filename is not None:
            folder = os.path.dirname(filename)

        # model results
        conf_mr = {}
        for name, mr in self.modelresults.items():
            conf_mr[name] = self._modelresult_to_dict(mr, folder)
        conf["modelresults"] = conf_mr

        # observations
        conf_obs = {}
        for name, obs in self.observations.items():
            conf_obs[name] = self._observation_to_dict(obs, folder)
        conf["observations"] = conf_obs

        if filename is not None:
            ext = os.path.splitext(filename)[-1]
            if (ext == ".yml") or (ext == ".yaml") or (ext == ".conf"):
                self._config_to_yml(filename, conf)
            elif "xls" in ext:
                self._config_to_excel(filename, conf)
            else:
                raise ValueError("Filename extension not supported! Use .yml or .xlsx")
        else:
            return conf

    @staticmethod
    def _modelresult_to_dict(mr, folder):
        d = {}
        # d["display_name"] = mr.name
        if mr.filename is None:
            raise ValueError(
                f"Cannot write Connector to conf file! ModelResult '{mr.name}' has no filename."
            )
        if folder is None:
            d["filename"] = mr.filename
        else:
            d["filename"] = os.path.relpath(mr.filename, start=folder)
        d["item"] = mr.item
        return d

    @staticmethod
    def _observation_to_dict(obs, folder):
        d = {}
        # d["display_name"] = obs.name
        d["type"] = obs.__class__.__name__
        if obs.filename is None:
            raise ValueError(
                f"Cannot write Connector to conf file! Observation '{obs.name}' has no filename."
            )
        if folder is None:
            d["filename"] = obs.filename
        else:
            d["filename"] = os.path.relpath(obs.filename, start=folder)
        d["item"] = obs._item
        if isinstance(obs, PointObservation):
            d["x"] = obs.x
            d["y"] = obs.y
        # d["variable_name"] = obs.variable_name
        return d

    @staticmethod
    def _config_to_excel(filename, conf):
        with pd.ExcelWriter(filename) as writer:
            dfmr = pd.DataFrame(conf["modelresults"]).T
            dfmr.index.name = "name"
            dfmr.to_excel(writer, sheet_name="modelresults")

            dfo = pd.DataFrame(conf["observations"]).T
            dfo.index.name = "name"
            dfo.to_excel(writer, sheet_name="observations")

            # dfo = pd.DataFrame(conf["connections"]).T
            # dfo.to_excel(writer, sheet_name="connections")

    @staticmethod
    def _config_to_yml(filename, conf):
        with open(filename, "w") as f:
            # TODO: preserve order
            yaml.dump(conf, f)  # , default_flow_style=False

    @staticmethod
    def from_config(conf: Union[dict, str], *, validate_eum=True, relative_path=True):
        """Load Connector from a config file (or dict)

        Parameters
        ----------
        configuration : Union[atr, dict]
            path to config file or dict with configuration
        validate_eum : bool, optional
            require eum to match, by default True
        relative_path: bool, optional
             file path are relative to configuration file, and not current directory

        Returns
        -------
        Connector
            A Connector object with the given configuration

        Examples
        --------
        >>> con = Connector.from_config('Oresund.yml')
        >>> cc = con.extract()
        """
        if isinstance(conf, str):
            filename = conf
            ext = os.path.splitext(filename)[-1]
            dirname = os.path.dirname(filename)
            if (ext == ".yml") or (ext == ".yaml") or (ext == ".conf"):
                conf = Connector._yaml_to_dict(filename)
            elif "xls" in ext:
                conf = Connector._excel_to_dict(filename)
            else:
                raise ValueError("Filename extension not supported! Use .yml or .xlsx")
        else:
            dirname = ""

        modelresults = {}
        for name, mr_dict in conf["modelresults"].items():
            if not mr_dict.get("include", True):
                continue
            if relative_path:
                filename = os.path.join(dirname, mr_dict["filename"])
            else:
                filename = mr_dict["filename"]
            item = mr_dict.get("item")
            mr = ModelResult(filename, name=name, item=item)
            modelresults[name] = mr
        mr_list = list(modelresults.values())

        observations = {}
        for name, obs_dict in conf["observations"].items():
            if not obs_dict.get("include", True):
                continue
            if relative_path:
                filename = os.path.join(dirname, obs_dict["filename"])
            else:
                filename = obs_dict["filename"]
            item = obs_dict.get("item")
            alt_name = obs_dict.get("name")
            name = name if alt_name is None else alt_name

            otype = obs_dict.get("type")
            if (otype is not None) and ("track" in otype.lower()):
                obs = TrackObservation(filename, item=item, name=name)
            else:
                x, y = obs_dict.get("x"), obs_dict.get("y")
                obs = PointObservation(filename, item=item, x=x, y=y, name=name)
            observations[name] = obs
        obs_list = list(observations.values())

        if "connections" in conf:
            raise NotImplementedError()
        else:
            con = Connector(obs_list, mr_list, validate=validate_eum)
        return con

    @staticmethod
    def _yaml_to_dict(filename):
        with open(filename) as f:
            contents = f.read()
        conf = yaml.load(contents, Loader=yaml.FullLoader)
        return conf

    @staticmethod
    def _excel_to_dict(filename):
        with pd.ExcelFile(filename, engine="openpyxl") as xls:
            dfmr = pd.read_excel(xls, "modelresults", index_col=0).T
            dfo = pd.read_excel(xls, "observations", index_col=0).T
            # try: dfc = pd.read_excel(xls, "connections", index_col=0).T
        conf = {}
        conf["modelresults"] = Connector._remove_keys_w_nan_value(dfmr.to_dict())
        conf["observations"] = Connector._remove_keys_w_nan_value(dfo.to_dict())
        return conf

    @staticmethod
    def _remove_keys_w_nan_value(d):
        """Loops through dicts in dict and removes all entries where value is NaN
        e.g. x,y values of TrackObservations
        """
        dout = {}
        for key, subdict in d.items():
            dout[key] = {k: v for k, v in subdict.items() if pd.Series(v).notna().all()}
        return dout
