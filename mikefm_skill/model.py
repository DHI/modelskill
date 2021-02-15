import os
import numpy as np
import warnings
from enum import Enum

from mikeio import Dfs0, Dfsu
from .observation import PointObservation, TrackObservation
from .compare import PointComparer, TrackComparer, ComparisonCollection


class ModelResultType(Enum):
    dfs0 = 0
    dfsu = 1
    dfs2 = 2


class ModelResult:
    name = None
    type = None
    filename = None
    dfs = None
    observations = None
    items = None
    start = None  # TODO: add start time

    def __init__(self, filename: str, name: str = None):
        # TODO: add "start" as user may wish to disregard start from comparison
        self.filename = filename
        ext = os.path.splitext(filename)[-1]
        if ext == ".dfsu":
            self.dfs = Dfsu(filename)
            self.type = ModelResultType.dfsu
        # elif ext == '.dfs2':
        #    self.dfs = Dfs2(filename)
        #    self.type = ModelResultType.dfs2
        elif ext == ".dfs0":
            self.dfs = Dfs0(filename)
            self.type = ModelResultType.dfs0
        else:
            raise ValueError(f"Filename extension {ext} not supported (dfsu, dfs0)")

        self.observations = {}

        if name is None:
            name = os.path.basename(filename).split(".")[0]
        self.name = name

    def __repr__(self):
        # return self.dfs
        out = []
        out.append("<mikefm_skill.ModelResult>")
        out.append(self.filename)
        return str.join("\n", out)

    def add_observation(self, observation, item):
        """Add an observation to this ModelResult

        Parameters
        ----------
        observation : <mikefm_skill.PointObservation>
            Observation object for later comparison
        item : str, integer
            ModelResult item name or number corresponding to the observation
        """
        ok = self._validate_observation(observation)
        if ok:
            observation.model_variable = item
            self.observations[observation.name] = observation
            # self.items.append(item)
        else:
            warnings.warn("Could not add observation")

    def _validate_observation(self, observation) -> bool:
        ok = False
        if self.type == ModelResultType.dfsu:
            if isinstance(observation, PointObservation):
                ok = self.dfs.contains([observation.x, observation.y])
            elif isinstance(observation, TrackObservation):
                ok = True
        elif self.type == ModelResultType.dfs0:
            # TODO: add check on name
            ok = True

        return ok

    # TODO: rename to compare() ?
    def extract(self) -> ComparisonCollection:
        """extract model result in all observations"""
        cc = ComparisonCollection()
        # for obs, item in zip(self.observations, self.items):
        for obs in self.observations.values():
            if isinstance(obs, PointObservation):
                comparison = self.compare_point_observation(obs, obs.model_variable)
            elif isinstance(obs, TrackObservation):
                comparison = self.compare_track_observation(obs, obs.model_variable)
            else:
                warnings.warn("Only point and track observation are supported!")
                continue

            cc.add_comparison(comparison)
        return cc

    def compare_point_observation(self, observation, item) -> PointComparer:
        """Compare this ModelResult with a point observation

        Parameters
        ----------
        observation : <mikefm_skill.PointObservation>
            Observation to be compared
        item : str, integer
            ModelResult item name or number

        Returns
        -------
        <mikefm_skill.PointComparer>
            A comparer object for further analysis or plotting
        """
        assert isinstance(observation, PointObservation)
        ds_model = None
        if self.type == ModelResultType.dfsu:
            ds_model = self._extract_point_dfsu(observation, item)
        elif self.type == ModelResultType.dfs0:
            ds_model = self._extract_point_dfs0(observation, item)

        return PointComparer(observation, ds_model)

    def _extract_point_dfsu(self, observation: PointObservation, item):
        assert isinstance(observation, PointObservation)
        xy = np.atleast_2d([observation.x, observation.y])
        elemids, _ = self.dfs.get_2d_interpolant(xy, n_nearest=1)
        ds_model = self.dfs.read(elements=elemids, items=[item])
        ds_model.items[0].name = self.name
        return ds_model

    def _extract_point_dfs0(self, observation, item):
        ds_model = self.dfs.read(items=[item])
        ds_model.items[0].name = self.name
        return ds_model

    def compare_track_observation(self, observation, item) -> TrackComparer:
        """Compare this ModelResult with a track observation

        Parameters
        ----------
        observation : <mikefm_skill.TrackObservation>
            Track observation to be compared
        item : str, integer
            ModelResult item name or number

        Returns
        -------
        <mikefm_skill.TrackComparer>
            A comparer object for further analysis or plotting
        """
        assert isinstance(observation, TrackObservation)
        ds_model = None
        if self.type == ModelResultType.dfsu:
            ds_model = self._extract_track_dfsu(observation, item)
        elif self.type == ModelResultType.dfs0:
            raise NotImplementedError()
            # ds_model = self._extract_track_dfs0(observation, item)

        return TrackComparer(observation, ds_model)

    def _extract_track_dfsu(self, observation: TrackObservation, item):
        assert isinstance(observation, TrackObservation)
        ds_model = self.dfs.extract_track(track=observation.df, items=[item])
        ds_model.items[-1].name = self.name
        return ds_model

    def plot_observation_positions(self, figsize=None):
        """Plot oberservation points on a map showing the model domain

        Parameters
        ----------
        figsize : (float, float), optional
            figure size, by default None
        """
        if self.type == ModelResultType.dfs0:
            warnings.warn(
                "Plotting observations is only supported for dfsu ModelResults"
            )
            return
        xn = self.dfs.node_coordinates[:, 0]
        offset_x = 0.02 * (max(xn) - min(xn))
        ax = self.dfs.plot(plot_type="outline_only", figsize=figsize)
        for obs in self.observations.values():
            if isinstance(obs, PointObservation):
                ax.scatter(x=obs.x, y=obs.y, marker="x")
                ax.annotate(obs.name, (obs.x + offset_x, obs.y))
            elif isinstance(obs, TrackObservation):
                ax.scatter(x=obs.x, y=obs.y, c=obs.values, marker=".", cmap="Reds")
        return ax


class ModelResultCollection:
    def __init__(self, modelresults=None):
        self.modelresults = {}
        if modelresults is not None:
            for mr in modelresults:
                self.add_modelresult(mr)

    def __repr__(self):
        out = []
        out.append(f"<{type(self).__name__}>")
        for key, value in self.modelresults.items():
            out.append(f"{type(value).__name__}: {key}")
        return str.join("\n", out)

    def __getitem__(self, x):
        return self.modelresults[x]

    def add_modelresult(self, modelresults):
        self.modelresults[modelresults.name] = modelresults

