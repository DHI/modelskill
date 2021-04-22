import os
from typing import Union
import numpy as np
import warnings

from abc import ABC, abstractmethod

from mikeio import Dfs0, Dfsu, Dataset, eum
from .observation import PointObservation, TrackObservation
from .compare import PointComparer, TrackComparer, ComparerCollection, BaseComparer
from .plot import plot_observation_positions


class ModelResultInterface(ABC):
    @abstractmethod
    def add_observation(self, observation, item, weight, validate_eum):
        pass

    @abstractmethod
    def extract(self) -> ComparerCollection:
        pass

    @abstractmethod
    def plot_observation_positions(self, figsize):
        pass


class ModelResult(ModelResultInterface):
    """
    The result from a MIKE FM simulation (either dfsu or dfs0)

    Examples
    --------
    >>> mr = ModelResult("Oresund2D.dfsu")

    >>> mr = ModelResult("Oresund2D_points.dfs0", name="Oresund")
    """

    def __init__(self, filename: str, name: str = None):
        # TODO: add "start" as user may wish to disregard start from comparison
        self.filename = filename
        ext = os.path.splitext(filename)[-1]
        if ext == ".dfsu":
            self.dfs = Dfsu(filename)
        # elif ext == '.dfs2':
        #    self.dfs = Dfs2(filename)
        elif ext == ".dfs0":
            self.dfs = Dfs0(filename)
        else:
            raise ValueError(f"Filename extension {ext} not supported (dfsu, dfs0)")

        self.observations = {}

        if name is None:
            name = os.path.basename(filename).split(".")[0]
        self.name = name

    def __repr__(self):
        out = []
        out.append("<fmskill.ModelResult>")
        out.append(self.filename)
        return "\n".join(out)

    def add_observation(self, observation, item, weight=1.0, validate_eum=True):
        """Add an observation to this ModelResult

        Parameters
        ----------
        observation : <fmskill.Observation>
            Observation object for later comparison
        item : str, integer
            ModelResult item name or number corresponding to the observation
        weight: float, optional
            Relative weight used in weighted skill calculation, default 1.0
        validate_eum: bool, optional
            Require eum type and units to match between model and observation?
            Defaut: True
        """
        ok = self._validate_observation(observation)
        if ok and validate_eum:
            ok = self._validate_item_eum(observation, item)
        if ok:
            observation.model_variable = item
            observation.weight = weight
            self.observations[observation.name] = observation
        else:
            warnings.warn("Could not add observation")

    def _validate_observation(self, observation) -> bool:
        ok = False
        if self.is_dfsu:
            if isinstance(observation, PointObservation):
                ok = self.dfs.contains([observation.x, observation.y])
            elif isinstance(observation, TrackObservation):
                ok = True
        elif self.is_dfs0:
            # TODO: add check on name
            ok = True
        return ok

    def _validate_item_eum(self, observation, mod_item) -> bool:
        """Check that observation and model item eum match"""
        ok = True
        obs_item = observation.itemInfo
        if obs_item.type == eum.EUMType.Undefined:
            warnings.warn(f"{observation.name}: Cannot validate as type is Undefined.")
            return ok

        mod_item = self._get_model_item(mod_item)
        if mod_item.type != obs_item.type:
            ok = False
            warnings.warn(
                f"{observation.name}: Item type should match. Model item: {mod_item.type.display_name}, obs item: {obs_item.type.display_name}"
            )
        if mod_item.unit != obs_item.unit:
            ok = False
            warnings.warn(
                f"{observation.name}: Unit should match. Model unit: {mod_item.unit.display_name}, obs unit: {obs_item.unit.display_name}"
            )
        return ok

    def _get_model_item(self, item) -> eum.ItemInfo:
        """"Given str or int find corresponding model itemInfo"""
        mod_items = self.dfs.items
        n_items = len(mod_items)
        if isinstance(item, int):
            if (item < 0) or (item >= n_items):
                raise ValueError(f"item number must be between 0 and {n_items}")
        elif isinstance(item, str):
            item_names = [i.name for i in mod_items]
            try:
                item = item_names.index(item)
            except ValueError:
                raise ValueError(f"item not found in model items ({item_names})")
        else:
            raise ValueError("item must be an integer or a string")
        return mod_items[item]

    def extract(self) -> ComparerCollection:
        """Extract model result in all observations"""
        cc = ComparerCollection()
        for obs in self.observations.values():
            comparer = self._extract_observation(obs, obs.model_variable)
            if comparer is not None:
                cc.add_comparer(comparer)
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
            comparer = PointComparer(observation, ds_model)
        elif isinstance(observation, TrackObservation):
            ds_model = self._extract_track(observation, item)
            comparer = TrackComparer(observation, ds_model)
        else:
            raise ValueError("Only point and track observation are supported!")

        return comparer

    def _extract_point(self, observation: PointObservation, item) -> Dataset:
        ds_model = None
        if self.is_dfsu:
            ds_model = self._extract_point_dfsu(observation.x, observation.y, item)
        elif self.is_dfs0:
            ds_model = self._extract_point_dfs0(item)

        return ds_model

    def _extract_point_dfsu(self, x, y, item):
        xy = np.atleast_2d([x, y])
        elemids, _ = self.dfs.get_2d_interpolant(xy, n_nearest=1)
        ds_model = self.dfs.read(elements=elemids, items=[item])
        ds_model.items[0].name = self.name
        return ds_model

    def _extract_point_dfs0(self, item):
        ds_model = self.dfs.read(items=[item])
        ds_model.items[0].name = self.name
        return ds_model

    def _extract_track(self, observation: TrackObservation, item) -> Dataset:
        ds_model = None
        if self.is_dfsu:
            ds_model = self._extract_track_dfsu(observation, item)
        elif self.is_dfs0:
            ds_model = self.dfs.read(items=[0, 1, item])
            ds_model.items[-1].name = self.name

        return ds_model

    def _extract_track_dfsu(self, observation: TrackObservation, item):
        ds_model = self.dfs.extract_track(track=observation.df, items=[item])
        ds_model.items[-1].name = self.name
        return ds_model

    def plot_observation_positions(self, figsize=None):
        """Plot observation points on a map showing the model domain

        Parameters
        ----------
        figsize : (float, float), optional
            figure size, by default None
        """
        if self.is_dfs0:
            warnings.warn(
                "Plotting observations is only supported for dfsu ModelResults"
            )
            return

        ax = plot_observation_positions(
            dfs=self.dfs, observations=self.observations.values()
        )

        return ax

    @property
    def is_dfsu(self):
        return isinstance(self.dfs, Dfsu)

    @property
    def is_dfs0(self):
        return isinstance(self.dfs, Dfs0)


class ModelResultCollection(ModelResultInterface):
    """
    A collection of results from multiple MIKE FM simulations
    with the same "topology", e.g. several "runs" of the same model.

    Examples
    --------
    >>> mr1 = ModelResult("HKZN_local_2017_v1.dfsu", name="HKZN_v1")
    >>> mr2 = ModelResult("HKZN_local_2017_v2.dfsu", name="HKZN_v2")
    >>> mr = ModelResultCollection([mr1, mr2])
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

    def add_observation(self, observation, item, weight=1.0, validate_eum=True):
        """Add an observation to all ModelResults in collection

        Parameters
        ----------
        observation : <fmskill.PointObservation>
            Observation object for later comparison
        item : str, integer
            ModelResult item name or number corresponding to the observation
        """
        for mr in self.modelresults.values():
            mr.add_observation(observation, item, weight, validate_eum)

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
            comparer = self._compare_point_observation(observation, item)
        elif isinstance(observation, TrackObservation):
            comparer = self._compare_track_observation(observation, item)
        else:
            raise ValueError("Only point and track observation are supported!")

        return comparer

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
            comparer = self._extract_observation(obs, obs.model_variable)
            if comparer is not None:
                cc.add_comparer(comparer)
        return cc

    def plot_observation_positions(self, figsize=None):
        """Plot observation points on a map showing the first model domain

        Parameters
        ----------
        figsize : (float, float), optional
            figure size, by default None
        """
        return self._mr0.plot_observation_positions(figsize=figsize)
