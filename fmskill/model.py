import os
from typing import Union
import numpy as np
import warnings
from enum import Enum

from mikeio import Dfs0, Dfsu, Dataset
from .observation import PointObservation, TrackObservation
from .compare import PointComparer, TrackComparer, ComparerCollection, BaseComparer


class ModelResultType(Enum):
    dfs0 = 0
    dfsu = 1
    dfs2 = 2


class ModelResult:
    # name = None
    # type = None
    # filename = None
    # dfs = None
    # observations = None
    # items = None
    # start = None  # TODO: add start time

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
        out.append("<fmskill.ModelResult>")
        out.append(self.filename)
        return str.join("\n", out)

    def add_observation(self, observation, item, weight=1.0):
        """Add an observation to this ModelResult

        Parameters
        ----------
        observation : <fmskill.PointObservation>
            Observation object for later comparison
        item : str, integer
            ModelResult item name or number corresponding to the observation
        weight: float
            Relative weight used in compound skill calculation, default 1.0
        """
        ok = self._validate_observation(observation)
        if ok:
            observation.model_variable = item
            observation.weight = weight
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

    def extract(self) -> ComparerCollection:
        """extract model result in all observations"""
        cc = ComparerCollection()
        for obs in self.observations.values():
            comparison = self._extract_observation(obs, obs.model_variable)
            if comparison is not None:
                cc.add_comparison(comparison)
        return cc

    def _extract_observation(
        self,
        observation: Union[PointObservation, TrackObservation],
        item: Union[int, str],
    ) -> BaseComparer:
        """Compare this ModelResult with an observation

        Parameters
        ----------
        observation : <PointObservation> or <TrackObservation>
            Observation to be compared
        item : str, integer
            ModelResult item name or number

        Returns
        -------
        <fmskill.BaseComparer>
            A comparer object for further analysis or plotting
        """
        if isinstance(observation, PointObservation):
            ds_model = self._extract_point(observation, item)
            comparison = PointComparer(observation, ds_model)
        elif isinstance(observation, TrackObservation):
            ds_model = self._extract_track(observation, item)
            comparison = TrackComparer(observation, ds_model)
        else:
            raise ValueError("Only point and track observation are supported!")

        return comparison

    def _extract_point(self, observation: PointObservation, item) -> Dataset:
        assert isinstance(observation, PointObservation)
        ds_model = None
        if self.type == ModelResultType.dfsu:
            ds_model = self._extract_point_dfsu(observation, item)
        elif self.type == ModelResultType.dfs0:
            ds_model = self._extract_point_dfs0(observation, item)
        return ds_model

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

    def _extract_track(self, observation: TrackObservation, item) -> Dataset:
        assert isinstance(observation, TrackObservation)
        ds_model = None
        if self.type == ModelResultType.dfsu:
            ds_model = self._extract_track_dfsu(observation, item)
        elif self.type == ModelResultType.dfs0:
            ds_model = self.dfs.read(items=[0, 1, item])
            ds_model.items[-1].name = self.name
        return ds_model

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
                if obs.n_points < 10000:
                    ax.scatter(x=obs.x, y=obs.y, c=obs.values, marker=".", cmap="Reds")
                else:
                    print("Too many points to plot")
                    # TODO: group by lonlat bin
        return ax


class ModelResultCollection:
    """Collection of ModelResult with same "topology"
    e.g. several "runs" of the same model. For calibration.
    Future: different type of models (local vs regional etc)
    """

    _mr0 = None

    @property
    def names(self):
        return list(self.modelresults.keys())

    @property
    def observations(self):
        return self._mr0.observations

    # has_same_topology = False

    def __init__(self, modelresults=None):
        self.modelresults = {}
        if modelresults is not None:
            for mr in modelresults:
                self.add_modelresult(mr)
        self._mr0 = self.modelresults[self.names[0]]

    def __repr__(self):
        out = []
        out.append(f"<{type(self).__name__}>")
        for key, value in self.modelresults.items():
            out.append(f"{type(value).__name__}: {key}")
        return str.join("\n", out)

    def __getitem__(self, x):
        return self.modelresults[x]

    def add_modelresult(self, modelresult):
        assert isinstance(modelresult, ModelResult)
        self.modelresults[modelresult.name] = modelresult

    def add_observation(self, observation, item):
        """Add an observation to all ModelResults in collection

        Parameters
        ----------
        observation : <fmskill.PointObservation>
            Observation object for later comparison
        item : str, integer
            ModelResult item name or number corresponding to the observation
        """
        for mr in self.modelresults.values():
            mr.add_observation(observation, item)

    def _extract_observation(
        self, observation: Union[PointComparer, TrackComparer], item: Union[int, str]
    ) -> BaseComparer:
        """Compare all ModelResults in collection with an observation

        Parameters
        ----------
        observation : <PointObservation> or <TrackObservation>
            Observation to be compared
        item : str, integer
            ModelResult item name or number

        Returns
        -------
        <fmskill.BaseComparer>
            A comparer object for further analysis or plotting
        """
        if isinstance(observation, PointObservation):
            comparison = self._compare_point_observation(observation, item)
        elif isinstance(observation, TrackObservation):
            comparison = self._compare_track_observation(observation, item)
        else:
            raise ValueError("Only point and track observation are supported!")

        return comparison

    def _compare_point_observation(self, observation, item) -> PointComparer:
        """Compare all ModelResults in collection with a point observation

        Parameters
        ----------
        observation : <fmskill.PointObservation>
            Observation to be compared
        item : str, integer
            ModelResult item name or number

        Returns
        -------
        <fmskill.PointComparer>
            A comparer object for further analysis or plotting
        """
        assert isinstance(observation, PointObservation)
        ds_model = []
        for mr in self.modelresults.values():
            ds_model.append(mr._extract_point(observation, item))

        return PointComparer(observation, ds_model)

    def _compare_track_observation(self, observation, item) -> TrackComparer:
        """Compare all ModelResults in collection with a track observation

        Parameters
        ----------
        observation : <fmskill.TrackObservation>
            Observation to be compared
        item : str, integer
            ModelResult item name or number

        Returns
        -------
        <fmskill.TrackComparer>
            A comparer object for further analysis or plotting
        """
        assert isinstance(observation, TrackObservation)
        ds_model = []
        for mr in self.modelresults.values():
            assert isinstance(mr, ModelResult)
            ds_model.append(mr._extract_track(observation, item))

        return TrackComparer(observation, ds_model)

    def extract(self) -> ComparerCollection:
        """extract model result in all observations"""
        cc = ComparerCollection()

        for obs in self.observations.values():
            comparison = self._extract_observation(obs, obs.model_variable)
            if comparison is not None:
                cc.add_comparison(comparison)

        # for mr in self.modelresults.values():
        #     for obs in mr.observations.values():
        #         comparison = mr.compare_observation(obs, obs.model_variable)
        #         if comparison is not None:
        #             comparison.name = comparison.name + "_" + mr.name
        #             cc.add_comparison(comparison)
        return cc

    def plot_observation_positions(self, figsize=None):
        """Plot observation points on a map showing the first model domain

        Parameters
        ----------
        figsize : (float, float), optional
            figure size, by default None
        """
        return self._mr0.plot_observation_positions(figsize=figsize)
