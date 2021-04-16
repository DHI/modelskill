import numpy as np


class SpatialSkill:
    @property
    def x(self):
        return self.ds.x

    @property
    def y(self):
        return self.ds.y

    @property
    def coords(self):
        return self.ds.coords

    @property
    def n(self):
        if "n" in self.ds:
            return self.ds.n

    def __init__(self, ds, name: str = None):
        self.ds = ds
        self.name = name

    def plot(self, field, **kwargs):
        return self.ds[field].plot(**kwargs)

    def to_dataframe(self):
        return self.ds.to_dataframe()
