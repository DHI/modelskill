import numpy as np
from mikeio import Dfs0, Dfsu

class ModelResult:
    filename = None
    dfs = None
    observations = None

    def __init__(self, filename):
        self.filename = filename
        self.dfs = Dfsu(filename)

    def __repr__(self):
        out = []
        out.append("ModelResult")
        out.append(self.filename)
        return str.join("\n", out)

    def add_observation(self, observation):
        pass

    def is_observation_in_domain(self, observation):
        pass

    def extract(self, observation):
        pass