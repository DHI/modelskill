from collections.abc import Mapping, Iterable, Sequence
from typing import List, Union
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from mikeio import Dfs0
from .model import ModelResult, ModelResultInterface
from .observation import Observation, PointObservation, TrackObservation
from .comparison import PointComparer, TrackComparer, ComparerCollection, BaseComparer


def compare(mod, obs):
    # con = SingleConnection(mod, obs)
    # return con.extract()
    if not isinstance(obs, Observation):
        obs = PointObservation(obs)
    if isinstance(mod, str):
        dfs = Dfs0(mod)
        if len(dfs.items) > 1:
            raise ValueError("Model ambiguous - please provide single item")
        mod = dfs.read().to_dataframe()
    elif isinstance(mod, pd.DataFrame):
        if len(mod.columns) > 1:
            raise ValueError("Model ambiguous - please provide single item")
    elif isinstance(mod, pd.Series):
        mod = mod.to_frame()
    c = PointComparer(obs, mod)
    return c


class SingleConnection:
    """A connection between model(s) and a single observation"""

    def __init__(self, mod=None, obs=None, validate=True):
        # mod_item, obs_item
        self.modelresults = self._parse_model(mod)
        self.obs = self._parse_observation(obs)
        self.name = self.obs.name
        if validate:
            self._validate()

    def _parse_model(self, mod) -> List[ModelResultInterface]:
        if isinstance(mod, Sequence) and (not isinstance(mod, str)):
            mr = []
            for m in mod:
                mr.append(self._parse_single_model(m))
        else:
            mr = [self._parse_single_model(mod)]
        return mr

    def _parse_single_model(self, mod) -> ModelResultInterface:
        if isinstance(mod, (pd.Series, pd.DataFrame)):
            return self._parse_pandas_model(mod)
        elif isinstance(mod, str):
            return self._parse_filename_model(mod)
        elif isinstance(mod, ModelResultInterface):
            return mod
        else:
            raise ValueError("Unknown model result type")

    def _parse_pandas_model(self, df) -> ModelResultInterface:
        return PointObservation(df)

    def _parse_filename_model(self, filename) -> ModelResultInterface:
        return

    def _parse_observation(self, obs) -> Observation:
        if isinstance(obs, (pd.Series, pd.DataFrame)):
            return PointObservation(obs)
        elif isinstance(obs, str):
            return PointObservation(obs)
        elif isinstance(obs, Observation):
            return obs
        else:
            raise ValueError("Unknown observation type")

    def _validate(self):
        pass

    def extract(self):
        pass


class Connector(Mapping, Sequence):
    """A Connector object can have multiple SingleConnections"""

    def __init__(self, mod=None, obs=None, validate=True):
        self.connections = {}
        if (mod is not None) and (obs is not None):
            self.add(mod, obs)
        elif (mod is not None) or (obs is not None):
            raise ValueError("mod and obs must both be specified (or both None)")

    def add(self, mod, obs, validate=True):
        con = SingleConnection(mod, obs, validate)
        self.connections[con.name] = con

    def extract(self) -> ComparerCollection:
        pass

    def to_config(self, filename: str):
        # write contents of connector to configuration file (yml or xlxs)
        raise NotImplementedError()

    @classmethod
    def from_config(cls, filename: str):
        # get connector from configuration file (yml or xlxs)
        raise NotImplementedError()
