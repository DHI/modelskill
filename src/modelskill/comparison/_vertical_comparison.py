from __future__ import annotations

import matplotlib
import numpy as np
from typing import TYPE_CHECKING, Callable, Iterable, Sequence, Tuple
from ..types import GeometryType

from ..plotting._misc import get_fig_ax
import xarray as xr
from matplotlib import dates as mdates
from ..model import PointModelResult, TrackModelResult
from ..model.network import NodeModelResult
from ..model.vertical import VerticalModelResult
from ._utils import parse_metric
from ..skill_profile import SkillProfile

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
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes to plot on (if None, a new figure and axes will be created).
        figsize : tuple, optional
            Size of the figure (only used if ax is None).

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the vertical profile plot.
        """
        from ._comparison import MOD_COLORS

        cmp = self.comparer
        _, ax = get_fig_ax(ax, figsize)

        title = title if title is not None else cmp.name

        obs_values = cmp.data["Observation"].values
        z_obs = cmp.data["z"].values

        for j in range(cmp.n_models):
            key = cmp.mod_names[j]
            raw_data = cmp.raw_mod_data[key].data
            mod_values = raw_data[key].values
            z = raw_data["z"].values
            ax.plot(mod_values, z, color=MOD_COLORS[j], label=key)

            if show_matched_model:
                mod_values_int = cmp.data[key].values
                ax.scatter(
                    mod_values_int,
                    z_obs,
                    color=MOD_COLORS[j],
                    marker=".",
                    label=key,
                )

        ax.scatter(
            obs_values,
            z_obs,
            color=cmp.data[cmp._obs_name].attrs["color"],
            marker=".",
            label=cmp._obs_name,
        )

        ax.set_xlabel(f"{cmp._unit_text}")
        ax.set_ylabel("z")
        ax.legend()
        ax.grid(True)
        ax.set_title(title)
        if self._pos_z():
            ax.invert_yaxis()
        return ax

    def hovmoller(
        self,
        *,
        title: str | None = None,
        model: str | int | None = None,
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
        model : str or int, optional
            Model to plot when multiple model results are present.
            If omitted and only one model is available, that model is used.
            If omitted and multiple models are available, a ValueError is raised.
        ylim : tuple, optional
            Limits for the depth axis (z).
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes to plot on (if None, a new figure and axes will be created).
        figsize : tuple, optional
            Size of the figure (only used if ax is None).
        obs_marker_size : float, optional
            Base observation marker size in points^2 for reference figure width.
        **kwargs
            Other keyword arguments to matplotlib's function contourf.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the Hovmöller plot.

        Examples
        -------
        >>> ax = cmp.vertical.plot.hovmoller(figsize=(16,5))
        >>> ax = cmp.vertical.plot.hovmoller(model="mod2", figsize=(16,5))
        """
        cmp = self.comparer
        mod_name = self._get_model_name(model)

        _, ax = get_fig_ax(ax, figsize)
        if title is None:
            title = f"{mod_name} and Observations"

        marker_size = 20 if obs_marker_size is None else obs_marker_size

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
        """True when z uses positive-down convention (deepest point has largest abs value)."""
        return abs(self.comparer.z.max()) > abs(self.comparer.z.min())

    def _get_model_name(self, model: str | int | None) -> str:
        cmp = self.comparer

        if model is None:
            if cmp.n_models == 1:
                return cmp.mod_names[0]
            raise ValueError(
                "Multiple models found. Please specify model by name or index."
            )

        if isinstance(model, int):
            if model < 0 or model >= cmp.n_models:
                raise IndexError(
                    f"Model index {model} out of range [0, {cmp.n_models - 1}]"
                )
            return cmp.mod_names[model]

        if model not in cmp.mod_names:
            raise KeyError(
                f"Unknown model '{model}'. Available models: {cmp.mod_names}"
            )

        return model


class VerticalAccessor:
    def __init__(self, comparer):
        self._comparer = comparer
        self.plot = VerticalPlotter(comparer)

    def _agg(self, agg_func: str = "mean"):
        """Aggregate the comparison vertically using a specified aggregation function."""
        from ._comparison import Comparer

        cmp = self._comparer

        grouped = cmp.data.groupby("time")
        ds = getattr(grouped, agg_func)()

        raw_mod_data: dict[
            str,
            PointModelResult | TrackModelResult | VerticalModelResult | NodeModelResult,
        ] = {}
        for mod_name in cmp.mod_names:
            r = cmp.raw_mod_data[mod_name].data
            r_grouped = r.where(
                (r.z >= cmp.z.min()) & (r.z <= cmp.z.max()), drop=True
            ).groupby("time")
            raw = getattr(r_grouped, agg_func)()
            raw_mod_data[mod_name] = PointModelResult(
                raw, x=cmp.x, y=cmp.y, quantity=cmp.quantity
            )

        ds.attrs = cmp.data.attrs
        ds.attrs["gtype"] = GeometryType.POINT
        ds.attrs["name"] = f"vertical_{agg_func}"

        return Comparer(ds, raw_mod_data=raw_mod_data)

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
        bins: int | Sequence | None = 5,
        metrics: Iterable[str] | Iterable[Callable] | str | Callable | None = None,
        n_min: int | None = None,
    ) -> SkillProfile:
        """Compute skill metrics as a function of depth using depth bins.

        This method computes metrics directly on binned long-format data (similar to
        `Comparer.gridded_skill`) and returns an xarray-backed `SkillProfile`.

        Parameters
        ----------
        bins : int or sequence
            Depth bins, passed directly to ``xarray.Dataset.groupby_bins``.
            See ``xarray.Dataset.groupby_bins(bins=...)`` for full details.

            - If int: number of equal-width bins.
            - If sequence: explicit bin edges.

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

        metric_funcs = parse_metric(metrics, directional=cmp.quantity.is_directional)

        def calculate_metric(g):
            obs = g[cmp._obs_name]
            mod_results = []
            for model_name in cmp.mod_names:
                mod = g[model_name]
                n = int(obs.count())

                if n_min is not None and n < n_min:
                    metrics_dict = {f.__name__: np.nan for f in metric_funcs}
                else:
                    metrics_dict = {
                        f.__name__: float(f(obs.values, mod.values))
                        for f in metric_funcs
                    }
                metrics_dict["n"] = n

                ds_metric = xr.Dataset(metrics_dict).expand_dims(model=[model_name])
                mod_results.append(ds_metric)

            return xr.concat(mod_results, dim="model")

        ds = cmp.data.groupby_bins("z", bins).map(calculate_metric)
        ds = ds.rename({"z_bins": "z"})
        ds = ds.drop_vars(["x", "y"], errors="ignore")
        if cmp.n_models == 1:
            ds = ds.squeeze("model", drop=True)

        return SkillProfile(ds)

    def _raw_model_to_z(self, raw_mod, z):
        df = raw_mod.data[[raw_mod.name]].to_dataframe()
        z_dist = (df["z"] - float(z)).abs()
        nearest_idx = (
            z_dist.reset_index().groupby("time", sort=False)["z"].idxmin().to_numpy()
        )
        sel_data = raw_mod.data.isel(time=np.sort(nearest_idx))
        return type(raw_mod)(sel_data)
