import numpy as np
from typing import Callable, Iterable, Sequence, Tuple
from ..types import GeometryType

from ..plotting._misc import _get_fig_ax
import xarray as xr
from matplotlib import dates as mdates
from ..model import PointModelResult


class VerticalPlotter:
    def __init__(self, comparer):
        self.comparer = comparer

    def profile(
        self,
        title: str | None = None,
        ax=None,
        figsize: Tuple[float, float] | None = None,
        **kwargs,
    ):
        """Plot a vertical profile of model and observations at a specific time.

        Parameters
        ----------
        title : str, optional
            Title of the plot. If None, the selected time will be used as the title.
        ax : matplotlib Axes, optional
            Matplotlib Axes to plot on (if None, a new figure and axes will be created).
        figsize : tuple, optional
            Size of the figure (only used if ax is None).
        **kwargs
            Other keyword arguments to matplotlib's plot function for the model and observation lines.

        Returns
        -------
        matplotlib Axes
            Axes object with the vertical profile plot.
        """
        cmp = self.comparer

        _, ax = _get_fig_ax(ax, figsize)
        avail_times = cmp.time.unique()

        for i, t in enumerate(avail_times):
            col = "C" + str(i)
            data = cmp.sel(time=t).data
            z = data["z"].values
            ax.plot(
                data[cmp.mod_names[0]].values,
                z,
                "o-",
                linewidth=2,
                label=f"{cmp.mod_names[0]}-t{i}",
                color=col,
            )
            ax.plot(
                data["Observation"].values,
                z,
                linestyle="--",
                linewidth=2,
                label=f"Observation-t{i}",
                color=col,
            )
            ax.set_xlabel(kwargs.get("xlabel", f"{cmp._unit_text}"))
            ax.set_ylabel(kwargs.get("ylabel", "Depth"))
            ax.legend()
            ax.grid(True)
            ax.set_title(title if title is not None else f"{t}")

        return ax

    def hovmoller(
        self,
        *,
        title: str | None = None,
        ylim: Tuple[float, float] | None = None,
        ax=None,
        figsize: Tuple[float, float] | None = None,
        obs_marker_size: float = 20.0,
        **kwargs,
    ):
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

        T = np.array(sorted(set(t)))
        n_layers = len(z) // len(T)  # assuming equal number of layers

        Z = z.reshape(len(T), n_layers)
        V = v.reshape(len(T), n_layers)
        T_grid, _ = np.meshgrid(T, np.arange(n_layers), indexing="ij")
        cf = ax.contourf(T_grid, Z, V, **kwargs)
        ax.figure.colorbar(cf, ax=ax)
        z_pos = False
        if z_pos:
            ax.invert_yaxis()

        # observations
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
                edgecolors="white",
                linewidths=1,
                vmin=V.min(),
                vmax=V.max(),
            )

        # Scale dates
        ax.set_xlim(T.min(), T.max())
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )

        ax.set_title(title)
        unit_text = cmp._unit_text
        ax.set_ylabel(kwargs.get("ylabel", f"Model, {unit_text}"))
        if ylim is not None:
            ax.set_ylim(ylim)

        return ax


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
        metrics: Iterable[str] | Iterable[Callable] | str | Callable | None = None,
    ):
        """
        Compute skill metrics as a function of depth (z).

        Parameters
        ----------
        metrics : list or None
            List of metric names or callables to compute (default: standard metrics).
        bins : int, sequence of tuple, or None
            Number of equal-width bins or sequence of depth bin edges as tuples (min, max). Mutually exclusive with binsize.
        binsize : float or None
            Bin size for depth (overrides bins if provided). Mutually exclusive with bins.

        Returns
        -------
        SkillProfile
            Skill metrics aggregated by depth.
        """
        from ._collection import ComparerCollection

        ds = self._comparer.data

        # Determine binning strategy
        zmin, zmax = float(ds["z"].min()), float(ds["z"].max())
        if binsize is not None:
            _bins = np.arange(zmin, zmax + binsize, binsize)
            _bins = [(bin, bin + binsize) for bin in _bins[:-1]]
        elif bins is not None:
            if isinstance(bins, int):
                binsize = (zmax - zmin) / bins
                _bins = np.arange(zmin, zmax + binsize, binsize)
                _bins = [(bin, bin + binsize) for bin in _bins[:-1]]
            else:
                _bins = bins
        else:
            return None

        cmps = []
        for b in _bins:
            name = f"{round(abs(b[0]), 2)}m-{round(abs(b[1]), 2)}m"
            cmps.append(
                self._comparer.where(
                    (self._comparer.data["z"] >= b[0])
                    & (self._comparer.data["z"] < b[1])
                ).rename({self._comparer.name: name})
            )
        return ComparerCollection(cmps).skill(metrics=metrics)
