import numpy as np


class SpatialSkill:
    def __init__(self, ds, name: str = None):
        self.ds = ds
        self.name = name

    def plot(self, **kwargs):
        return self.ds.plot(**kwargs)
