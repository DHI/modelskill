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
    def obs_names(self):
        if "observation" in self._coords_list:
            return self.coords["observation"].values
        else:
            return []

    @property
    def mod_names(self):
        if "model" in self._coords_list:
            return self.coords["model"].values
        else:
            return []

    @property
    def field_names(self):
        return list(self.ds.data_vars)

    @property
    def _coords_list(self):
        return [d for d in self.coords.dims]

    @property
    def n(self):
        if "n" in self.ds:
            return self.ds.n

    def __init__(self, ds, name: str = None):
        self.ds = ds
        self.name = name

    def plot(self, field, model=None, **kwargs):
        if field not in self.field_names:
            raise ValueError(f"field {field} not found in {self.field_names}")

        if model is None:
            ds = self.ds[field]
        else:
            if model not in self.mod_names:
                raise ValueError(f"model {model} not in model list ({self.mod_names})")
            ds = self.ds[field].sel({"model": model})

        extra_dims = [d for d in ds.coords.dims if d not in ["x", "y"]]
        if len(extra_dims) == 2:
            ax = ds.plot(col=extra_dims[0], row=extra_dims[1], **kwargs)
        if len(extra_dims) == 1:
            ax = ds.plot(col=extra_dims[0], **kwargs)
        else:
            ax = ds.plot(**kwargs)
        return ax

    def to_dataframe(self):
        return self.ds.to_dataframe()

    def __repr__(self):
        return repr(self.ds)