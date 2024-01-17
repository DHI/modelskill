from __future__ import annotations
import os

from typing import Optional, Sequence
import warnings
import numpy as np
import pandas as pd

import mikeio
import yaml

from . import model_result, plotting
from .matching import _single_obs_compare
from .obs import Observation, PointObservation
from .utils import is_iterable_not_str
from .comparison import Comparer, ComparerCollection


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
    return d


def _config_to_excel(filename, conf):
    with pd.ExcelWriter(filename) as writer:
        dfmr = pd.DataFrame(conf["modelresults"]).T
        dfmr.index.name = "name"
        dfmr.to_excel(writer, sheet_name="modelresults")

        dfo = pd.DataFrame(conf["observations"]).T
        dfo.index.name = "name"
        dfo.to_excel(writer, sheet_name="observations")


def _config_to_yml(filename, conf):
    with open(filename, "w") as f:
        # TODO: preserve order
        yaml.dump(conf, f)  # , default_flow_style=False


class SingleObsConnector:
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
        # deprecated
        warnings.warn(
            FutureWarning(
                "SingleObsConnector is deprecated! Use `modelskill.match` instead"
            )
        )
        self.obs = None
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

    def _parse_model(self, mod):
        if is_iterable_not_str(mod):
            mr = []
            for m in mod:
                mr.append(self._parse_single_model(m))
        else:
            mr = [self._parse_single_model(mod)]
        return mr

    def _parse_single_model(self, mod):
        if isinstance(mod, (pd.Series, pd.DataFrame, mikeio.DataArray)):
            return model_result(mod)
        else:
            return mod

    def _validate(self, obs, modelresults):
        # TODO: add validation errors to list
        ok = True
        for mod in modelresults:
            # has_mod_item = self._has_mod_item(mod)
            quantity_match = obs.quantity.is_compatible(mod.quantity)
            time_overlaps = self._validate_start_end(obs, mod)
            ok = ok and quantity_match and time_overlaps
        return ok

    @staticmethod
    def _validate_start_end(obs, mod):
        if obs.time[-1] < mod.time[0]:
            warnings.warn(
                f"No time overlap! Obs '{obs.name}' end is before model '{mod.name}' start"
            )
            return False

        if obs.time[0] > mod.time[-1]:
            warnings.warn(
                f"No time overlap! Obs '{obs.name}' start is after model '{mod.name}' end"
            )
            return False

        return True

    @property
    def n_models(self):
        """Number of (unique) model results in Connector."""
        return len(self.modelresults)

    @property
    def mod_names(self):
        """Names of (unique) model results in Connector."""
        return list(self.modelresults.keys())

    def plot_observation_positions(self, figsize=None):
        """Plot observation points on a map showing the model domain

        Parameters
        ----------
        figsize : (float, float), optional
            figure size, by default None
        """
        return plotting.spatial_overview(
            obs=[self.obs], mod=self.modelresults, figsize=figsize
        )

    def _parse_observation(self, obs):
        if isinstance(obs, Observation):
            return obs
        elif isinstance(obs, (pd.Series, pd.DataFrame)):
            return PointObservation(obs)
        elif isinstance(obs, str):
            return PointObservation(obs)
        else:
            raise ValueError(f"Unknown observation type {type(obs)}")

    def extract(self, **kwargs) -> Optional[Comparer]:
        """Extract model results at times and positions of observation.

        Returns
        -------
        Comparer
            A comparer object for further analysis and plotting.
        """
        comparer = _single_obs_compare(obs=self.obs, mod=self.modelresults, **kwargs)
        if comparer.n_points == 0:
            warnings.warn(f"No overlapping data was found for {comparer.name}!")
            return None
        return comparer


class Connector(Sequence):
    @property
    def n_observations(self):
        """Number of (unique) observations in Connector."""
        return len(self.observations)

    @property
    def obs_names(self):
        """Names of (unique) observations in Connector."""
        return list(self.observations.keys())

    @property
    def n_models(self):
        """Number of (unique) model results in Connector."""
        return len(self.modelresults)

    @property
    def mod_names(self):
        """Names of (unique) model results in Connector."""
        return list(self.modelresults.keys())

    def __repr__(self):
        txt = "<Connector> with \n"
        return txt + "\n".join(" -" + repr(c) for c in self.connections.values())

    def __init__(self, obs=None, mod=None, weight=1.0, validate=True):
        warnings.warn(
            FutureWarning("Connector is deprecated! Use `modelskill.match` instead")
        )

        self.name = None
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
        >>> mr = DfsuModelResult("Oresund2D.dfsu", item=0)
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
        elif isinstance(obs, SingleObsConnector):
            con = obs
        elif isinstance(obs, Observation):
            con = SingleObsConnector(obs, mod, weight=weight, validate=validate)
        else:
            raise ValueError(f"Unknown observation type {type(obs)}")
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
        return self.obs_names[self._get_obs_idx(obs)]

    def _get_obs_idx(self, obs):
        n_con = len(self.connections)
        if obs is None or n_con <= 1:
            return 0
        elif isinstance(obs, str):
            if obs in self.obs_names:
                obs_idx = self.obs_names.index(obs)
            else:
                raise KeyError(
                    f"connection {obs} could not be found in {self.obs_names}"
                )
        elif isinstance(obs, int):
            if obs < 0:  # Handle negative indices
                obs += n_con
            if obs >= 0 and obs < n_con:
                obs_idx = obs
            else:
                raise IndexError(f"connection id {obs} is out of range (0, {n_con-1})")
        else:
            raise KeyError("connection must be None, str or int")
        return obs_idx

    def __getitem__(self, x):
        if isinstance(x, int):
            x = self._get_obs_name(x)

        return self.connections[x]

    def __len__(self) -> int:
        return len(self.connections)

    def extract(self, *args, **kwargs) -> ComparerCollection:
        """Extract model results at times and positions of all observations.

        Returns
        -------
        ComparerCollection
            A comparer object for further analysis and plotting.
        """

        cmps = [con.extract(*args, **kwargs) for con in self.connections.values()]
        return ComparerCollection(cmps)

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
        return plotting.spatial_overview(obs=obs, mod=mod, title=title, figsize=figsize)

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
        obs = list(self.observations.values())
        mods = list(self.modelresults.values()) if show_model else []
        return plotting.temporal_coverage(
            obs=obs,
            mod=mods,
            limit_to_model_period=limit_to_model_period,
            title=title,
            figsize=figsize,
            marker=marker,
        )

    def to_config(self, filename: Optional[str] = None, relative_path=True):
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
            conf_mr[name] = _modelresult_to_dict(mr, folder)
        conf["modelresults"] = conf_mr

        # observations
        conf_obs = {}
        for name, obs in self.observations.items():
            conf_obs[name] = _observation_to_dict(obs, folder)
        conf["observations"] = conf_obs

        if filename is not None:
            ext = os.path.splitext(filename)[-1]
            if (ext == ".yml") or (ext == ".yaml") or (ext == ".conf"):
                _config_to_yml(filename, conf)
            elif "xls" in ext:
                _config_to_excel(filename, conf)
            else:
                raise ValueError("Filename extension not supported! Use .yml or .xlsx")
        else:
            return conf
