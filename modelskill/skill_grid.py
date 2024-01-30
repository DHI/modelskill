from __future__ import annotations
from typing import Any, Iterable, overload, Hashable, TYPE_CHECKING
import warnings
import xarray as xr

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    import pandas as pd


class SkillGridMixin:
    data: xr.DataArray | xr.Dataset

    @property
    def x(self) -> xr.DataArray:
        """x-coordinate values"""
        return self.data.x  # type: ignore

    @property
    def y(self) -> xr.DataArray:
        """y-coordinate values"""
        return self.data.y  # type: ignore

    @property
    def coords(self) -> Any:
        """Coordinates (same as xr.DataSet.coords)"""
        return self.data.coords

    @property
    def obs_names(self) -> list[str]:
        """List of observation names"""
        if "observation" in self._coords_list:
            return list(self.coords["observation"].values)
        else:
            return []

    @property
    def mod_names(self) -> list[str]:
        """List of model names"""
        if "model" in self._coords_list:
            return list(self.coords["model"].values)
        else:
            return []

    @property
    def _coords_list(self) -> list[str]:
        return [str(d) for d in self.coords.dims]


class SkillGridArray(SkillGridMixin):
    """A SkillGridArray is a single metric-SkillGrid, corresponding to a "column" in a SkillGrid

    Typically created by indexing a SkillGrid object, e.g. `ss["bias"]`.

    Examples
    --------
    >>> gs = cc.gridded_skill()
    >>> gs["bias"].plot()
    """

    def __init__(self, data: xr.DataArray) -> None:
        assert isinstance(data, xr.DataArray)
        self.data = data

    def __repr__(self) -> str:
        out = [
            "<SkillGridArray>",
            f"Dimensions: (x: {len(self.x)}, y: {len(self.y)})",
        ]
        return "\n".join(out)

    def plot(self, model: str | None = None, **kwargs: Any) -> Axes:
        """wrapper for xArray DataArray plot function

        Parameters
        ----------
        model : str, optional
            Name of model to plot, by default all models
        **kwargs
            keyword arguments passed to xr.DataArray plot()
            e.g. figsize

        Examples
        --------
        >>> gs = cc.gridded_skill()
        >>> gs["bias"].plot()
        >>> gs.rmse.plot(model='SW_1')
        >>> gs.r2.plot(cmap='YlOrRd', figsize=(10,10))
        """
        if model is None:
            da = self.data
        else:
            warnings.warn(
                "model argument is deprecated, use sel(model=...)",
                FutureWarning,
            )
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
    >>> gs = cc.gridded_skill()
    >>> gs.metrics
    ['n', 'bias', 'rmse', 'urmse', 'mae', 'cc', 'si', 'r2']

    >>> gs.mod_names
    ['SW_1', 'SW_2']

    >>> gs.sel(model='SW_1').rmse.plot()
    """

    def __init__(self, data: xr.Dataset) -> None:
        # TODO: add type and unit info; add domain to plot outline on map
        self.data = data
        self._set_attrs()

    @property
    def metrics(self) -> list[str]:
        """List of metrics (=data vars)"""
        return list(self.data.data_vars)

    def __repr__(self) -> str:
        out = [
            "<SkillGrid>",
            f"Dimensions: (x: {len(self.x)}, y: {len(self.y)})",
        ]
        return "\n".join(out)

    @overload
    def __getitem__(self, key: Hashable) -> SkillGridArray:
        ...

    @overload
    def __getitem__(self, key: Iterable[Hashable]) -> SkillGrid:
        ...

    def __getitem__(
        self, key: Hashable | Iterable[Hashable]
    ) -> SkillGridArray | SkillGrid:
        result = self.data[key]
        if isinstance(result, xr.DataArray):
            return SkillGridArray(result)
        elif isinstance(result, xr.Dataset):
            return SkillGrid(result)
        else:
            return result

    def __getattr__(self, item: str, *args, **kwargs) -> Any:
        if item in self.data.data_vars:
            return self[item]  # Redirects to __getitem__
        else:
            # return getattr(self.data, item, *args, **kwargs)
            raise AttributeError(
                f"""
                    SkillGrid has no attribute {item}; Maybe you are
                    looking for the corresponding xr.Dataset attribute?
                    Access SkillGrid's Dataset with '.data'.
                """
            )

    def _set_attrs(self) -> None:
        # TODO: use type and unit to give better long name
        # self.ds["bias"].attrs = dict(long_name="Bias of Hm0", units="m")

        self.data["n"].attrs = dict(long_name="Number of observations", units="-")
        if self._has_geographical_coords():
            self.data["x"].attrs = dict(long_name="Longitude", units="degrees east")
            self.data["y"].attrs = dict(long_name="Latitude", units="degrees north")
        else:
            self.data["x"].attrs = dict(long_name="Easting", units="meter")
            self.data["y"].attrs = dict(long_name="Northing", units="meter")

    def _has_geographical_coords(self) -> bool:
        is_geo = True
        if (self.x.min() < -180.0) or (self.x.max() > 360.0):
            is_geo = False
        if (self.y.min() < -90.0) or (self.y.max() > 90.0):
            is_geo = False
        return is_geo

    def sel(self, model: str) -> SkillGrid:
        """Select a model from the SkillGrid

        Parameters
        ----------
        model : str
            Name of model to select

        Returns
        -------
        SkillGrid
            SkillGrid with only the selected model
        """
        sel_data = self.data.sel(model=model)
        assert isinstance(sel_data, xr.Dataset)
        return SkillGrid(sel_data)

    def plot(self, metric: str, model: str | None = None, **kwargs: Any) -> Axes:
        warnings.warn(
            "plot() is deprecated and will be removed in a future version. ",
            FutureWarning,
        )
        if metric not in self.metrics:
            raise ValueError(f"metric {metric} not found in {self.metrics}")
        return self[metric].plot(model=model, **kwargs)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert gridded skill data to pandas DataFrame

        Returns
        -------
        pd.DataFrame
            data as a pandas DataFrame
        """
        return self.data.to_dataframe()
