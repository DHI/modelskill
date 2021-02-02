import numpy as np
import warnings
from mikeio import Dfs0, Dfsu
from mikefm_skill.observation import PointObservation

class ModelResult:
    filename = None
    dfs = None
    observations = []
    items = []

    def __init__(self, filename):
        # add "start" as user may wish to disregard start from comparison
        self.filename = filename
        self.dfs = Dfsu(filename)

    def __repr__(self):
        #return self.dfs
        out = []
        out.append("ModelResult")
        out.append(self.filename)
        return str.join("\n", out)

    def add_observation(self, observation, item):
        ok = self._validate_observation(observation)
        if ok:
            self.observations.append(observation)
            self.items.append(item)
        else:
            warnings.warn('Could not add observation')

    def _validate_observation(self, observation):
        return self.dfs.contains([observation.x, observation.y])

    def extract(self):
        """extract model result in observation positions
        """
        ex = []
        for j, obs in enumerate(self.observations):
            mrp = self._extract_point_observation(obs, self.items[j])
            ex.append(mrp)
        return ex

    def _extract_point_observation(self, observation, item):
        elemids, _ = self.dfs.get_2d_interpolant([observation.x, observation.y], n_nearest=1)
        modeldata = self.dfs.read(elements=elemids, items=[item])
        return ModelResultPoint(observation, modeldata)



class ModelResultPoint:
    observation = None
    model = None
    time = None
    data = None

    def __init__(self, observation, data):
        self.observation = observation
        self.data = data

    def plot(self):
        pass

    def statistics(self):
        pass