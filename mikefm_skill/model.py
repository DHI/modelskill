import numpy as np
import warnings
from mikeio import Dfs0, Dfsu
from mikefm_skill.observation import PointObservation

class ModelResult:
    filename = None
    dfs = None
    observations = []

    def __init__(self, filename):
        self.filename = filename
        self.dfs = Dfsu(filename)

    def __repr__(self):
        out = []
        out.append("ModelResult")
        out.append(self.filename)
        return str.join("\n", out)

    def add_observation(self, observation):
        ok = self._validate_observation(observation)
        if ok:
            self.observations.append(observation)
        else:
            warnings.warn('Could not add observation')

    def _validate_observation(self, observation):
        return self.dfs.contains([observation.x, observation.y])

    def extract(self):
        pass