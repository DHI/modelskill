import numpy as np
from typing import Callable, Iterable, Sequence, Tuple
from ..types import GeometryType

from ..plotting._misc import _get_fig_ax
import xarray as xr
from matplotlib import dates as mdates
from ..model import PointModelResult, TrackModelResult, VerticalModelResult


class VerticalPlotter:
    def __init__(self, comparer):
        self.comparer = comparer

    def timeseries(self, z_model: float | None = None, ax=None, figsize=None, **kwargs):
        """
        Plot a timeseries of model and observation data at a specific depth.

        Parameters
        ----------
        z_model : float, optional
            Depth at which to plot the timeseries. If None, all available depths will be used.
        ax : matplotlib Axes, optional
            Matplotlib Axes to plot on (if None, a new figure and axes will be created).
        figsize : tuple, optional
            Size of the figure (only used if ax is None).
        **kwargs
            Additional keyword arguments to pass to the plotting function.

        Returns
        -------
        matplotlib Axes
            Axes object with the timeseries plot.

        Example usage:
        -------
        >>> cmp.vertical.plot.timeseries(z_model=-5)
        >>> cmp.vertical.plot.timeseries()  # Try to match model z from comparison data
        """
        _, ax = _get_fig_ax(ax, figsize)

        cmp = self.comparer
        if z_model is None:
            cmp_z = cmp.data["z"].values
            if np.isnan(cmp_z):
                raise ValueError(
                    "No z_model provided and no 'z' coordinate found in comparison data."
                )
        else:
            cmp_z = z_model

        if cmp.gtype != "point":
            raise ValueError(
                "Timeseries plot is only available for point comparisons. Use slice() to create a point comparison at a specific depth."
            )
        if "z" not in cmp.data.coords:
            raise ValueError(
                "Comparison data must have a 'z' coordinate for vertical plotting."
            )

        mod_name = self.comparer.mod_names[0]

        d = cmp.raw_mod_data[mod_name].data
        cmp.data["Observation"].plot(
            marker="o", linestyle="", label="Observation", ax=ax
        )
        # Find nearest model depth layer to requested z and plot that timeseries.
        z_layers = np.unique(d.z.values)
        nearest_z = z_layers[np.argmin(np.abs(z_layers - cmp_z))]

        d.where((d.z == nearest_z), drop=True)[mod_name].plot(
            label=f"{mod_name} (z={nearest_z:g})", ax=ax
        )
        return ax

    def profile(
        self,
        time: np.datetime64 | str | int | None = None,
        method: str = "exact",
        title: str | None = None,
        ax=None,
        figsize: Tuple[float, float] | None = None,
        **kwargs,
    ):
        """Plot a vertical profile of model and observations at a specific time.

        Parameters
        ----------
        time : np.datetime64, str, int, or None
            Time to plot. Can be a datetime string, a numpy datetime64, or an integer index. If None, the first available time will be used.
        method : str, optional
            Method to handle time selection when an exact match is not found. Options are 'exact'
            (default) which raises an error if no exact match is found, or 'nearest' which selects the nearest available time.
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

        Example usage:
        -------
        >>> ax = cmp.vertical.plot.profile(time="2022-06-13 12:00", method="nearest")
        >>> ax = cmp.vertical.plot.profile(time=0)  # Plot the first available time
        """
        cmp = self.comparer
        avail_times = cmp.time.unique()

        _, ax = _get_fig_ax(ax, figsize)

        if time is None:
            sel_time = avail_times[0]

        if isinstance(time, int):
            if time < 0 or time >= len(avail_times):
                raise ValueError(
                    f"Time index {time} is out of bounds for available times."
                )
            else:
                sel_time = avail_times[time]
        else:
            sel_time = np.datetime64(time)
            if sel_time not in avail_times and method == "nearest":
                sel_time = avail_times[np.argmin(np.abs(avail_times - sel_time))]
            else:
                raise ValueError(
                    f"Time {sel_time} not an available time. Try using method='nearest'"
                )

        data = cmp.data.sel(time=sel_time)
        z = data["z"].values

        ax.plot(
            data[cmp.mod_names[0]].values, z, "o-", linewidth=2, label=cmp.mod_names[0]
        )
        ax.plot(data["Observation"].values, z, "o-", linewidth=2, label="Observation")
        ax.set_xlabel(kwargs.get("xlabel", f"{cmp._unit_text}"))
        ax.set_ylabel(kwargs.get("ylabel", "Depth"))
        ax.legend()
        ax.grid(True)
        ax.set_title(title if title is not None else f"{sel_time}")

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
        ax.set_xlabel(kwargs.get("xlabel", f"Observation, {unit_text}"))
        ax.set_ylabel(kwargs.get("ylabel", f"Model, {unit_text}"))
        if ylim is not None:
            ax.set_ylim(ylim)

        return ax


class VerticalAccessor:
    def __init__(self, comparer):
        self._comparer = comparer
        self.plot = VerticalPlotter(comparer)

    def _raw_model_at_nearest_z(
        self, z: float
    ) -> dict[str, PointModelResult | TrackModelResult | VerticalModelResult]:
        cmp = self._comparer
        mod_name = cmp.mod_names[0]
        d = cmp.raw_mod_data[mod_name].data
        z_layers = np.unique(d.z.values)
        nearest_z = z_layers[np.argmin(np.abs(z_layers - z))]
        new_raw = d.where((d.z == nearest_z), drop=True)
        raw_pm = PointModelResult(
            new_raw,
            x=1,
            y=1,
            z=nearest_z,
            quantity=cmp.quantity,
        )
        return {mod_name: raw_pm}

    def _raw_model_for_agg(
        self, agg_func: str
    ) -> dict[str, PointModelResult | TrackModelResult | VerticalModelResult]:
        cmp = self._comparer
        mod_name = cmp.mod_names[0]

        d = cmp.raw_mod_data[mod_name].data
        d.where((d.z >= cmp.z.min()) & (d.z <= cmp.z.max()), drop=True).groupby(
            "time"
        ).mean()

        # mod_df = d_raw[[mod_name, "z"]].to_dataframe()
        # obs_bounds = (
        #     cmp.data[["z"]]
        #     .to_dataframe()
        #     .groupby(level="time")["z"]
        #     .agg(["min", "max"])
        # )
        # bounded = mod_df.join(obs_bounds, how="inner")
        # bounded = bounded[(bounded["z"] >= bounded["min"]) & (bounded["z"] <= bounded["max"])]

        # if bounded.empty:
        #     raise ValueError("No raw model data within observed depth range.")

        # if agg_func == "mean":
        #     mod = bounded.groupby(level="time")[mod_name].mean()
        # elif agg_func == "min":
        #     mod = bounded.groupby(level="time")[mod_name].min()
        # elif agg_func == "max":
        #     mod = bounded.groupby(level="time")[mod_name].max()
        # else:
        #     raise ValueError(f"Unsupported aggregation function: {agg_func}")

        # mod = mod.to_xarray()

        # raw_pm = PointModelResult(
        #     mod.to_dataset(name=mod_name),
        #     x=1,
        #     y=1,
        #     quantity=cmp.quantity,
        # )
        return {mod_name: raw_pm}

    def slice(self, z: float, name: str = "slice"):
        from ._comparison import Comparer

        cmp = self._comparer
        cmp_out = cmp.where(cmp.data["z"] == z).rename({cmp.name: name})
        cmp_out.data.attrs["gtype"] = GeometryType.POINT
        cmp_out.data["z"] = z

        return Comparer(cmp_out.data, raw_mod_data=self._raw_model_at_nearest_z(z))

    def cut(
        self,
        zmin: float | None = None,
        zmax: float | None = None,
        name: str = "cut",
    ):
        """Cut the comparison to a specific depth range."""
        z = self._comparer.data["z"]
        z_minimum = float(z.min())
        z_maximum = float(z.max())

        zmin = zmin if zmin is not None else z_minimum
        zmax = zmax if zmax is not None else z_maximum

        cmp = self._comparer
        return cmp.where((cmp.data["z"] >= zmin) & (cmp.data["z"] <= zmax)).rename(
            {cmp.name: name}
        )

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

        raw["z"] = 0
        raw = raw.set_coords("z")

        ds = xr.Dataset({"Observation": obs, cmp.mod_names[0]: mod})
        ds.attrs = cmp.data.attrs
        ds.attrs["gtype"] = GeometryType.POINT
        ds.attrs["name"] = f"vertical_{agg_func}"
        ds["z"] = 0
        ds = ds.set_coords("z")

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
