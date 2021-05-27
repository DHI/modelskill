from collections.abc import Mapping, Iterable, Sequence
from fmskill.plot import plot_observation_positions
from typing import List, Union
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from mikeio import Dfs0, eum
from .model import (
    ModelResult,
    ModelResultInterface,
    DataFrameModelResult,
    ModelResultCollection,
)
from .observation import Observation, PointObservation, TrackObservation
from .comparison import PointComparer, TrackComparer, ComparerCollection, BaseComparer
from .utils import is_iterable_not_str


def compare(obs, mod, mod_item=None):
    # return SingleConnection(obs, mod).extract()
    if not isinstance(obs, Observation):
        obs = PointObservation(obs)

    mod = _parse_model(mod, mod_item)
    return PointComparer(obs, mod)


def _parse_model(mod, item=None):
    if isinstance(mod, str):
        dfs = Dfs0(mod)
        if (len(dfs.items) > 1) and (item is None):
            raise ValueError("Model ambiguous - please provide item")
        mod = dfs.read().to_dataframe()
    elif isinstance(mod, pd.DataFrame):
        if len(mod.columns) > 1:
            raise ValueError("Model ambiguous - please provide item")
    elif isinstance(mod, pd.Series):
        mod = mod.to_frame()
    elif isinstance(mod, ModelResult):
        if not mod.is_dfs0:
            raise ValueError("Only dfs0 ModelResults are supported")
        if mod.item is None:
            raise ValueError("Model ambiguous - please provide item")
        mod = mod._extract_point_dfs0(mod.item).to_dataframe()
    return mod


class BaseConnector:
    def __init__(self):
        self.modelresults = {}

    @property
    def n_models(self):
        return len(self.modelresults)

    @property
    def mod_names(self):
        return list(self.modelresults.keys())


class SingleObsConnector(BaseConnector):
    """A connection between a single observation and model(s)"""

    @property
    def _mrc(self):
        if self.n_models == 1:
            return self.modelresults[0]
        else:
            return ModelResultCollection(self.modelresults)

    def __repr__(self):
        obs_txt = f"obs={self.name}(n={self.obs.n_points})"
        mod_txt = f"model={self.modelresults[0].name}"
        if self.n_models > 1:
            mod_txt = f"{self.n_models} models="
            mod_txt += "[" + ", ".join(m.name for m in self.modelresults) + "]"
            if len(mod_txt) > 25:
                mod_txt = mod_txt[:20] + "...]"

        return f"<SingleConnector> {obs_txt} :: {mod_txt}"

    def __init__(self, obs, mod, mod_item=None, validate=True):
        # mod_item is temporary solution
        self.modelresults = self._parse_model(mod, mod_item)
        self.obs = self._parse_observation(obs)
        self.name = self.obs.name

        ok = self._validate()
        if validate and (not ok):
            raise ValueError("Validation failed! Cannot connect observation and model.")

    def _parse_model(self, mod, item=None) -> List[ModelResultInterface]:
        if is_iterable_not_str(mod):
            mr = []
            for m in mod:
                mr.append(self._parse_single_model(m, item))
        else:
            mr = [self._parse_single_model(mod, item)]
        return mr

    def _parse_single_model(self, mod, item=None) -> ModelResultInterface:
        if isinstance(mod, (pd.Series, pd.DataFrame)):
            return self._parse_pandas_model(mod, item)
        elif isinstance(mod, str):
            return self._parse_filename_model(mod, item)
        elif isinstance(mod, ModelResultInterface):
            # if mod.item is None:
            if item is not None:
                mod.item = item
            return mod
        else:
            raise ValueError(f"Unknown model result type {type(mod)}")

    def _parse_pandas_model(self, df, item=None) -> ModelResultInterface:
        return DataFrameModelResult(df, item=item)

    def _parse_filename_model(self, filename, item=None) -> ModelResultInterface:
        return ModelResult(filename, item=item)

    def _parse_observation(self, obs) -> Observation:
        if isinstance(obs, (pd.Series, pd.DataFrame)):
            return PointObservation(obs)
        elif isinstance(obs, str):
            return PointObservation(obs)
        elif isinstance(obs, Observation):
            return obs
        else:
            raise ValueError(f"Unknown observation type {type(obs)}")

    def _validate(self):
        # TODO: add validation errors to list
        ok = True
        for mod in self.modelresults:
            eum_match = self._validate_eum(self.obs, mod)
            in_domain = True
            if isinstance(mod, ModelResult) and isinstance(self.obs, PointObservation):
                in_domain = mod._in_domain(self.obs.x, self.obs.y)
                if not in_domain:
                    warnings.warn(
                        f"Obs '{self.obs.name}' outside domain of model '{mod.name}'"
                    )
            time_overlaps = self._validate_start_end(self.obs, mod)
            ok = ok and eum_match and in_domain and time_overlaps
        return ok

    @staticmethod
    def _validate_eum(obs, mod):
        """Check that observation and model item eum match"""
        assert isinstance(obs, Observation)
        assert isinstance(mod, ModelResultInterface)
        ok = True
        has_eum = lambda x: (x.itemInfo is not None) and (
            x.itemInfo.type != eum.EUMType.Undefined
        )

        # we can only check if both have eum itemInfo
        if has_eum(obs) and has_eum(mod):
            if obs.itemInfo.type != mod.itemInfo.type:
                ok = False
                warnings.warn(
                    f"Item type should match. Obs '{obs.name}' item: {obs.itemInfo.type.display_name}, model '{mod.name}' item: {mod.itemInfo.type.display_name}"
                )
            if obs.itemInfo.unit != mod.itemInfo.unit:
                ok = False
                warnings.warn(
                    f"Unit should match. Obs '{obs.name}' unit: {obs.itemInfo.unit.display_name}, model '{mod.name}' unit: {mod.itemInfo.unit.display_name}"
                )
        return ok

    @staticmethod
    def _validate_start_end(obs, mod):
        try:
            # need to put this in try-catch due to error in dfs0 in mikeio
            if obs.end_time < mod.start_time:
                warnings.warn(
                    f"Obs '{obs.name}' end is before model '{mod.name}' start"
                )
                return False
            if obs.start_time > mod.end_time:
                warnings.warn(f"Obs '{obs.name}' start is after model '{mod.name}' end")
                return False
        except:
            pass
        return True

    def plot_observation_positions(self, figsize=None):
        """Plot observation points on a map showing the model domain

        Parameters
        ----------
        figsize : (float, float), optional
            figure size, by default None
        """
        mod = self.modelresults[0]

        if mod.is_dfs0:
            warnings.warn(
                "Plotting observations is only supported for dfsu ModelResults"
            )
            return

        ax = plot_observation_positions(dfs=mod.dfs, observations=[self.obs])

        return ax

    def extract(self) -> BaseComparer:
        return self._mrc.extract_observation(self.obs, validate=False)


class Connector(BaseConnector, Mapping, Sequence):
    """A Connector object can have multiple SingleConnectors"""

    @property
    def n_observations(self):
        return len(self.observations)

    @property
    def obs_names(self):
        return list(self.observations.keys())

    def __repr__(self):
        txt = "<Connector> with \n"
        return txt + "\n".join(" -" + repr(c) for c in self.connections.values())

    def __init__(self, obs=None, mod=None, mod_item=None, validate=True):
        self.connections = {}
        self.observations = {}
        self.modelresults = {}
        if (mod is not None) and (obs is not None):
            if not is_iterable_not_str(obs):
                obs = [obs]
            for o in obs:
                self.add(o, mod, mod_item=mod_item, validate=validate)
        elif (mod is not None) or (obs is not None):
            raise ValueError("obs and mod must both be specified (or both None)")

    def add(self, obs, mod, mod_item=None, validate=True):
        if is_iterable_not_str(obs):
            for o in obs:
                self.add(o, mod, mod_item=mod_item, validate=validate)
            return
        elif isinstance(obs, SingleObsConnector):
            con = obs
        else:
            con = SingleObsConnector(obs, mod, mod_item=mod_item, validate=validate)
        self.connections[con.name] = con
        self._add_observation(con.obs)
        self._add_modelresults(con.modelresults)

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
            if obs >= 0 and obs < n_con:
                obs_id = obs
            else:
                raise IndexError(
                    f"connection id was {obs} - must be within 0 and {n_con-1}"
                )
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

    def extract(self) -> ComparerCollection:
        """do extract for all connections"""
        cc = ComparerCollection()

        for con in self.connections.values():
            comparer = con.extract()
            if comparer is not None:
                cc.add_comparer(comparer)
        return cc

    def plot_observation_positions(self, figsize=None):
        """Plot observation points on a map showing the model domain

        Parameters
        ----------
        figsize : (float, float), optional
            figure size, by default None
        """
        mod = list(self.modelresults.values())[0]

        if mod.is_dfs0:
            warnings.warn(
                "Plotting observations is only supported for dfsu ModelResults"
            )
            return

        observations = list(self.observations.values())
        ax = plot_observation_positions(dfs=mod.dfs, observations=observations)
        return ax

    def plot_temporal_coverage(self, limit_to_model_period=True):

        # TODO: multiple model
        mod0 = list(self.modelresults.values())[0]

        fig, ax = plt.subplots()
        y = np.repeat(0.0, 2)
        x = mod0.start_time, mod0.end_time
        plt.plot(x, y)
        labels = ["Model"]

        plt.plot([mod0.start_time, mod0.end_time], y)
        for key, obs in self.observations.items():
            y += 1.0
            plt.plot(obs.time, y[0] * np.ones_like(obs.values), "_", markersize=5)
            labels.append(key)
        if limit_to_model_period:
            plt.xlim([mod0.start_time, mod0.end_time])

        plt.yticks(np.arange(0, len(self.observations) + 1), labels)
        fig.autofmt_xdate()
        return ax

    def to_config(self, filename: str):
        # write contents of connector to configuration file (yml or xlxs)
        raise NotImplementedError()

    @classmethod
    def from_config(cls, filename: str):
        # get connector from configuration file (yml or xlxs)
        raise NotImplementedError()
