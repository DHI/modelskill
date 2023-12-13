from __future__ import annotations
from typing import Iterable, Optional, overload, Hashable
import warnings
import xarray as xr


class SkillGridMixin:
    @property
    def x(self):
        """x-coordinate values"""
        return self.data.x

    @property
    def y(self):
        """y-coordinate values"""
        return self.data.y

    @property
    def coords(self):
        """Coordinates (same as xr.DataSet.coords)"""
        return self.data.coords

    @property
    def obs_names(self):
        """List of observation names"""
        if "observation" in self._coords_list:
            return list(self.coords["observation"].values)
        else:
            return []

    @property
    def mod_names(self):
        """List of model names"""
        if "model" in self._coords_list:
            return list(self.coords["model"].values)
        else:
            return []

    @property
    def _coords_list(self):
        return [d for d in self.coords.dims]


class SkillGridArray(SkillGridMixin):
    """A SkillGridArray is a single metric-SkillGrid, corresponding to a "column" in a SkillGrid

    Typically created by indexing a SkillGrid object, e.g. `ss["bias"]`.

    Examples
    --------
    >>> ss = cc.gridded_skill()
    >>> ss["bias"].plot()
    """

    def __init__(self, data):
        assert isinstance(data, xr.DataArray)
        self.data = data

    def __repr__(self):
        return repr(self.data)

    def _repr_html_(self):
        return self.data._repr_html_()

    def plot(self, model=None, **kwargs):
        """wrapper for xArray DataArray plot function

        Parameters
        ----------
        model : str, optional
            Name of model to plot, by default all models
        **kwargs : keyword arguments passed to xr.DataArray plot()
            e.g. figsize

        Examples
        --------
        >>> ss = cc.gridded_skill()
        >>> ss["bias"].plot()
        >>> ss.rmse.plot(model='SW_1')
        >>> ss.r2.plot(cmap='YlOrRd', figsize=(10,10))
        """
        if model is None:
            da = self.data
        else:
            if model not in self.mod_names:
                raise ValueError(f"model {model} not in model list ({self.mod_names})")
            da = self.data.sel({"model": model})

        extra_dims = [d for d in da.coords.dims if d not in ["x", "y"]]
        if len(extra_dims) == 2:
            ax = da.plot(col=extra_dims[0], row=extra_dims[1], **kwargs)
        elif len(extra_dims) == 1:
            ax = da.plot(col=extra_dims[0], **kwargs)
        else:
            ax = da.plot(**kwargs)
        return ax


class SkillGrid(SkillGridMixin):
    """
    Gridded skill object for analysis and visualization of spatially
    gridded skill data. The object wraps the xr.DataSet class
    which can be accessed from the attribute data.

    The object contains one or more "arrays" of skill metrics, each
    corresponding to a single metric (e.g. bias, rmse, r2). The arrays
    are indexed by the metric name, e.g. `ss["bias"]` or `ss.bias`.

    Examples
    --------
    >>> ss = cc.gridded_skill()
    >>> ss.field_names
    ['n', 'bias', 'rmse', 'urmse', 'mae', 'cc', 'si', 'r2']

    >>> ss.mod_names
    ['SW_1', 'SW_2']

    >>> ss.rmse.plot(model='SW_1')
    """

    def __init__(self, data, name: Optional[str] = None):
        # TODO: add type and unit info; add domain to plot outline on map
        self.data = data
        self.name = name
        self._set_attrs()

    @property
    def field_names(self):
        # TODO: rename to metrics? (be consistent with Skill class)
        """List of field names (=data vars)"""
        return list(self.data.data_vars)

    def __repr__(self):
        return repr(self.data)

    def _repr_html_(self):
        return self.data._repr_html_()

    @overload
    def __getitem__(self, key: Hashable | int) -> SkillGridArray:
        ...

    @overload
    def __getitem__(self, key: Iterable[Hashable]) -> SkillGrid:
        ...

    def __getitem__(self, key) -> SkillGridArray | SkillGrid:
        result = self.data[key]
        if isinstance(result, xr.DataArray):
            return SkillGridArray(result)
        elif isinstance(result, xr.Dataset):
            return SkillGrid(result)
        else:
            return result

    def __getattr__(self, item):
        if item in self.data.data_vars:
            return self[item]  # Redirects to __getitem__

        # For other attributes, return them directly
        return getattr(self.data, item)

    def _set_attrs(self):
        # TODO: use type and unit to give better long name
        # self.ds["bias"].attrs = dict(long_name="Bias of Hm0", units="m")

        self.data["n"].attrs = dict(long_name="Number of observations", units="-")
        if self._has_geographical_coords():
            self.data["x"].attrs = dict(long_name="Longitude", units="degrees east")
            self.data["y"].attrs = dict(long_name="Latitude", units="degrees north")
        else:
            self.data["x"].attrs = dict(long_name="Easting", units="meter")
            self.data["y"].attrs = dict(long_name="Northing", units="meter")

    def _has_geographical_coords(self):
        is_geo = True
        if (self.x.min() < -180.0) or (self.x.max() > 360.0):
            is_geo = False
        if (self.y.min() < -90.0) or (self.y.max() > 90.0):
            is_geo = False
        return is_geo

    def plot(self, field: str, model=None, **kwargs):
        warnings.warn(
            "plot() is deprecated and will be removed in a future version. ",
            FutureWarning,
        )
        if field not in self.field_names:
            raise ValueError(f"field {field} not found in {self.field_names}")
        return self[field].plot(model=model, **kwargs)

    def to_dataframe(self):
        """export as pandas.DataFrame"""
        return self.data.to_dataframe()