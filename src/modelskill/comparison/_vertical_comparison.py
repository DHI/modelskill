from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Callable, Iterable, Sequence, Tuple
from ..types import GeometryType

from ..plotting._misc import _get_fig_ax
import xarray as xr
from matplotlib import dates as mdates
from ..model import PointModelResult
from ..metrics import _parse_metric
from ..skill_profile import SkillProfile
from ._utils import _groupby_df, _parse_groupby

if TYPE_CHECKING:
    import matplotlib.axes


class VerticalPlotter:
    def __init__(self, comparer):
        self.comparer = comparer

    def __call__(self, *args, **kwargs) -> matplotlib.axes.Axes:
        """Plot scatter plot of modelled vs observed data"""
        return self.profile(*args, **kwargs)

    def profile(
        self,
        title: str | None = None,
        ax: matplotlib.axes.Axes | None = None,
        figsize: Tuple[float, float] | None = None,
        show_matched_model: bool = False,
    ) -> matplotlib.axes.Axes:
        """Plot vertical profiles of model and observations.

        Parameters
        ----------
        title : str, optional
            Title of the plot.
            ax : matplotlib Axes, optional
            Matplotlib Axes to plot on (if None, a new figure and axes will be created).
        figsize : tuple, optional
            Size of the figure (only used if ax is None).

        Returns
        -------
        matplotlib Axes
            Axes object with the vertical profile plot.
        """
        from ._comparison import MOD_COLORS

        cmp = self.comparer
        _, ax = _get_fig_ax(ax, figsize)

        title = title if title is not None else cmp.name

        for j in range(cmp.n_models):
            key = cmp.mod_names[j]
            raw_data = cmp.raw_mod_data[key].data
            mod_values = raw_data[key].values
            z = raw_data["z"].values
            ax.plot(mod_values, z, color=MOD_COLORS[j])

            obs_values = cmp.data["Observation"].values
            z_obs = cmp.data["z"].values

            ax.scatter(
                obs_values,
                z_obs,
                color=cmp.data[cmp._obs_name].attrs["color"],
                marker=".",
            )

            if show_matched_model:
                mod_values_int = cmp.data[key].values
                ax.scatter(
                    mod_values_int,
                    z_obs,
                    color=MOD_COLORS[j],
                    marker=".",
                )

        ax.set_xlabel(f"{cmp._unit_text}")
        ax.set_ylabel("z")
        ax.legend([*cmp.mod_names, cmp._obs_name])
        ax.grid(True)
        ax.set_title(title)
        if self._pos_z():
            ax.invert_yaxis()
        return ax

    def hovmoller(
        self,
        *,
        title: str | None = None,
        ylim: Tuple[float, float] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        figsize: Tuple[float, float] | None = (12, 4),
        obs_marker_size: float = 20.0,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """
        Hovmöller plot of model and observations as a function of depth and time.

        Parameters
        ----------
        title : str, optional
            Title of the plot.
        ylim : tuple, optional
            Limits for the depth axis (z).
        ax : matplotlib Axes, optional
            Matplotlib Axes to plot on (if None, a new figure and axes will be created).
        figsize : tuple, optional
            Size of the figure (only used if ax is None).
        obs_marker_size : float, optional
            Base observation marker size in points^2 for reference figure width.
        **kwargs
            Other keyword arguments to matplotlib's function contourf.

        Returns
        -------
        matplotlib Axes
            Axes object with the Hovmöller plot.

        Example usage:
        -------
        >>> ax = cmp.vertical.plot.hovmoller(figsize=(16,5))
        """
        _, ax = _get_fig_ax(ax, figsize)
        if title is None:
            title = f"Hovmöller Plot of {self.comparer.mod_names[0]} vs Observations"

        marker_size = 20 if obs_marker_size is None else obs_marker_size

        cmp = self.comparer
        mod_name = cmp.mod_names[0]
        mod_at_obs = cmp.raw_mod_data[mod_name].values
        z = cmp.raw_mod_data[mod_name].z
        t = cmp.raw_mod_data[mod_name].time
        v = mod_at_obs

        T = np.unique(t)
        n_times = T.size

        if n_times == 0:
            raise ValueError(
                "No timesteps found in vertical profile data (t is empty)."
            )

        n_layers, remainder = divmod(len(z), n_times)
        if remainder:
            raise ValueError(
                "Inconsistent vertical profile data: expected equal number of z layers per time step. "
                f"Got len(z)={len(z)} and len(unique time)={n_times}."
            )

        Z = z.reshape(len(T), n_layers)
        V = v.reshape(len(T), n_layers)
        T_grid, _ = np.meshgrid(T, np.arange(n_layers), indexing="ij")
        cf = ax.contourf(T_grid, Z, V, **kwargs)
        cbar = ax.figure.colorbar(cf, ax=ax)
        cbar.set_label(cmp._unit_text)
        if self._pos_z():
            ax.invert_yaxis()

        # # Overlay observation points
        obs = cmp.data["Observation"]
        obs_t = set(sorted(obs.time.values))

        for date in obs_t:
            obs_on_date = obs[obs.time == date]
            ax.scatter(
                [date] * len(obs_on_date),
                obs_on_date["z"],
                c=obs_on_date.values,
                s=marker_size,
                edgecolors="none",
                linewidths=1,
                vmin=V.min(),
                vmax=V.max(),
                marker="s",
            )

            ax.scatter(
                [date] * len(obs_on_date),
                obs_on_date["z"],
                c=obs_on_date.values,
                s=10,
                edgecolors="white",
                linewidths=1,
                vmin=V.min(),
                vmax=V.max(),
                marker=".",
            )

        # Scale dates
        ax.set_xlim(T.min(), T.max())
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )

        ax.set_title(title)
        ax.set_ylabel("z")
        if ylim is not None:
            ax.set_ylim(ylim)

        return ax

    def _pos_z(self):
        return abs(self.comparer.z.max()) > abs(self.comparer.z.min())


class VerticalAccessor:
    def __init__(self, comparer):
        self._comparer = comparer
        self.plot = VerticalPlotter(comparer)

    def _agg(self, agg_func: str = "mean"):
        """Aggregate the comparison vertically using a specified aggregation function."""
        from ._comparison import Comparer

        cmp = self._comparer

        mod_name = cmp.mod_names[0]
        r = cmp.raw_mod_data[mod_name].data
        r_grouped = r.where(
            (r.z >= cmp.z.min()) & (r.z <= cmp.z.max()), drop=True
        ).groupby("time")

        if agg_func == "mean":
            obs = cmp.data.Observation.groupby("time").mean()
            mod = cmp.data[cmp.mod_names[0]].groupby("time").mean()
            raw = r_grouped.mean()
        elif agg_func == "min":
            obs = cmp.data.Observation.groupby("time").min()
            mod = cmp.data[cmp.mod_names[0]].groupby("time").min()
            raw = r_grouped.min()
        elif agg_func == "max":
            obs = cmp.data.Observation.groupby("time").max()
            mod = cmp.data[cmp.mod_names[0]].groupby("time").max()
            raw = r_grouped.max()
        else:
            raise ValueError(f"Unsupported aggregation function: {agg_func}")

        ds = xr.Dataset({"Observation": obs, cmp.mod_names[0]: mod})
        ds.attrs = cmp.data.attrs
        ds.attrs["gtype"] = GeometryType.POINT
        ds.attrs["name"] = f"vertical_{agg_func}"

        return Comparer(
            ds,
            raw_mod_data={
                mod_name: PointModelResult(
                    raw, x=cmp.x, y=cmp.y, z=0, quantity=cmp.quantity
                )
            },
        )

    def mean(self):
        """Aggregate the comparison vertically using a specified aggregation function."""
        return self._agg(agg_func="mean")

    def min(self):
        """Aggregate the comparison vertically using a specified aggregation function."""
        return self._agg(agg_func="min")

    def max(self):
        """Aggregate the comparison vertically using a specified aggregation function."""
        return self._agg(agg_func="max")

    def skill(
        self,
        bins: int | Sequence[Tuple[float, float]] | None = 5,
        binsize: float | None = None,
        by: str | Iterable[str] | None = None,
        metrics: Iterable[str] | Iterable[Callable] | str | Callable | None = None,
        n_min: int | None = None,
    ) -> SkillProfile:
        """Compute skill metrics as a function of depth using depth bins.

        This method computes metrics directly on binned long-format data (similar to
        `Comparer.gridded_skill`) and returns an xarray-backed `SkillProfile`.

        Parameters
        ----------
        bins : int, sequence of tuple, or None, optional
            Number of equal-width bins or sequence of explicit depth intervals as
            tuples (min, max). If `None`, returns `None`.
        binsize : float, optional
            Bin size for depth. If provided, overrides `bins`.
        by : str or iterable of str, optional
            Additional group-by fields, e.g. ``"freq:M"``.
        metrics : list, str, callable, optional
            Metrics to compute.
        n_min : int, optional
            Minimum number of observations in a depth bin; bins with fewer
            observations get metric values of `np.nan`.

        Returns
        -------
        SkillProfile
            Depth-binned skill metrics.
        """
        cmp = self._comparer
        if cmp.n_points == 0:
            raise ValueError("No data selected for skill assessment")

        z_bins = self._resolve_z_bins(cmp.data, bins=bins, binsize=binsize)
        if len(z_bins) <= 1:
            raise ValueError(
                "Only one depth bin found, skill profile requires multiple bins. "
                "Adjust 'bins' or 'binsize' parameters or use skill() "
                "method on Comparer directly for overall skill."
            )

        metric_funcs = _parse_metric(metrics, directional=cmp.quantity.is_directional)

        df = cmp._to_long_dataframe()

        df = self._add_depth_bins_to_df(df=df, z_bins=z_bins)

        agg_cols = _parse_groupby(by=by, n_mod=cmp.n_models, n_qnt=1)
        if "z" not in agg_cols:
            agg_cols.insert(0, "z")

        df = df.drop(columns=["z"]).rename(columns={"zBin": "z"})
        res = _groupby_df(df, by=agg_cols, metrics=metric_funcs, n_min=n_min)

        ds = res.to_xarray().squeeze()

        return SkillProfile(ds)

    @staticmethod
    def _resolve_z_bins(
        ds: xr.Dataset,
        *,
        bins: int | Sequence[Tuple[float, float]] | None,
        binsize: float | None,
    ) -> list[tuple[float, float]]:
        zmin, zmax = float(ds["z"].min()), float(ds["z"].max())

        if binsize is not None:
            edges = np.arange(zmin, zmax + binsize, binsize)
            return [(float(edge), float(edge + binsize)) for edge in edges[:-1]]

        if bins is None:
            raise ValueError("bins cannot be None")

        if isinstance(bins, int):
            binsize = (zmax - zmin) / bins
            edges = np.arange(zmin, zmax + binsize, binsize)
            return [(float(edge), float(edge + binsize)) for edge in edges[:-1]]

        return [(float(lo), float(hi)) for lo, hi in bins]

    @staticmethod
    def _add_depth_bins_to_df(
        *, df: pd.DataFrame, z_bins: Sequence[tuple[float, float]]
    ) -> pd.DataFrame:
        labels = [f"{round(abs(lo), 2)}-{round(abs(hi), 2)}" for lo, hi in z_bins]

        zbin = pd.Series(
            pd.Categorical([np.nan] * len(df), categories=labels, ordered=True),
            index=df.index,
        )
        last_idx = len(z_bins) - 1
        for i, ((lo, hi), label) in enumerate(zip(z_bins, labels)):
            # Make the last bin right-inclusive to ensure full coverage of max depth.
            if i == last_idx:
                mask = (df["z"] >= lo) & (df["z"] <= hi)
            else:
                mask = (df["z"] >= lo) & (df["z"] < hi)
            zbin = zbin.where(~mask, label)

        df = df.copy()
        df["zBin"] = zbin
        return df

    def _raw_model_to_z(self, raw_mod, z):
        df = raw_mod.data[[raw_mod.name]].to_dataframe()
        z_dist = (df["z"] - float(z)).abs()
        nearest_idx = (
            z_dist.reset_index().groupby("time", sort=False)["z"].idxmin().to_numpy()
        )
        sel_data = raw_mod.data.isel(time=np.sort(nearest_idx))
        return type(raw_mod)(sel_data)

    def _raw_model_to_fixed_z(self, raw_mod, z):
        z_layers = np.unique(raw_mod.data["z"].values)
        nearest = z_layers[np.argmin(np.abs(z_layers - float(z)))]
        sel_data = raw_mod.data.where(raw_mod.data["z"] == nearest, drop=True)
        return type(raw_mod)(sel_data)
