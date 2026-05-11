from __future__ import annotations

from typing import Any, Sequence, TYPE_CHECKING, overload

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    import pandas as pd


class SkillProfileMixin:
    data: xr.DataArray | xr.Dataset

    @property
    def z(self) -> xr.DataArray:
        """Depth-coordinate values."""
        return self.data.z  # type: ignore

    @property
    def coords(self) -> Any:
        """Coordinates (same as xr.DataSet.coords)."""
        return self.data.coords

    @property
    def obs_names(self) -> list[str]:
        """List of observation names."""
        if "observation" in self._coords_list:
            return list(self.coords["observation"].values)
        return []

    @property
    def mod_names(self) -> list[str]:
        """List of model names."""
        if "model" in self._coords_list:
            return list(self.coords["model"].values)
        return []

    @property
    def _coords_list(self) -> list[str]:
        return [str(d) for d in self.coords.dims]


class SkillProfileArray(SkillProfileMixin):
    """A single metric from a SkillProfile."""

    def __init__(self, data: xr.DataArray) -> None:
        assert isinstance(data, xr.DataArray)
        self.data = data

    def __repr__(self) -> str:
        return f"<SkillProfileArray>\nDimensions: (z: {len(self.z)})"

    def plot(self, **kwargs: Any) -> Axes:
        """Horizontal bar plot of metric values by depth bin."""
        da = self.data

        # Horizontal bar plot for categorical z
        ax = plt.gca()
        z_labels = [str(z) for z in da["z"].values]
        z_pos = np.arange(len(z_labels))

        # Reduce to one numeric value per z-bin for bar widths.
        z_first = da.transpose("z", ...)
        values = np.asarray(z_first.values, dtype=float)
        widths = values.reshape(len(z_labels), -1).mean(axis=1)

        # Create horizontal bar plot with error bars
        ax.barh(z_pos, widths, capsize=5, **kwargs)
        ax.set_yticks(z_pos)
        ax.set_yticklabels(z_labels)
        ax.set_ylabel("z")
        ax.set_xlabel(str(da.name) if da.name is not None else "value")

        return ax


class SkillProfile(SkillProfileMixin):
    """Skill metrics aggregated in depth bins."""

    def __init__(self, data: xr.Dataset) -> None:
        self.data = data
        self._set_attrs()

    @property
    def metrics(self) -> Sequence[Any]:
        """List of metrics (=data vars)."""
        return list(self.data.data_vars)

    def __repr__(self) -> str:
        return f"<SkillProfile>\nDimensions: (z: {len(self.z)})"

    @overload
    def __getitem__(self, key: str) -> SkillProfileArray: ...

    @overload
    def __getitem__(self, key: list[str]) -> SkillProfile: ...

    def __getitem__(self, key: str | list[str]) -> SkillProfileArray | SkillProfile:
        result = self.data[key]
        if isinstance(result, xr.DataArray):
            return SkillProfileArray(result)
        if isinstance(result, xr.Dataset):
            return SkillProfile(result)
        return result

    def __getattr__(self, item: str, *args, **kwargs) -> Any:
        if item in self.data.data_vars:
            return self[item]
        raise AttributeError(
            f"""
                SkillProfile has no attribute {item}; maybe you are looking for the
                corresponding xr.Dataset attribute? Access SkillProfile's Dataset with '.data'.
            """
        )

    def _set_attrs(self) -> None:
        if "n" in self.data:
            self.data["n"].attrs = dict(long_name="Number of observations", units="-")
        self.data["z"].attrs = dict(long_name="Depth bin", units="-")

    def sel(self, z: str) -> SkillProfile:
        """Select a depth bin from the SkillProfile."""
        sel_data = self.data.sel(z=z)
        if isinstance(sel_data, xr.DataArray):
            sel_data = sel_data.to_dataset(name=sel_data.name or "value")
        assert isinstance(sel_data, xr.Dataset)
        return SkillProfile(sel_data)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert skill profile data to pandas DataFrame."""
        return self.data.to_dataframe()
