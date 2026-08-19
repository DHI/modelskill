from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

import matplotlib.colors as colors
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from ._collection import ComparerCollection

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .. import metrics as mtr
from ..plotting import TaylorPoint, scatter, taylor_diagram, _plotly
from ..plotting._backend import (
    Backend,
    reject_matplotlib_axes,
    validate_backend,
)
from ..plotting._misc import _get_fig_ax, _xtick_directional, _ytick_directional
from ..settings import options
from ..utils import _get_idx
from ._comparer_plotter import quantiles_xy


def _default_univarate_title(kind: str, cc: ComparerCollection) -> str:
    return f"{kind} for {cc.n_observations} observations"


class ComparerCollectionPlotter:
    """Plotter for ComparerCollection

    Examples
    --------
    >>> cc.plot.scatter()
    >>> cc.plot.hist()
    >>> cc.plot.kde()
    >>> cc.plot.taylor()
    >>> cc.plot.box()
    """

    def __init__(self, cc: ComparerCollection) -> None:
        self.cc = cc
        self.is_directional = False

    def __call__(self, *args: Any, **kwds: Any) -> Axes | list[Axes]:
        return self.scatter(*args, **kwds)

    def scatter(
        self,
        *,
        bins: int | float = 120,
        quantiles: int | Sequence[float] | None = None,
        fit_to_quantiles: bool = False,
        show_points: bool | int | float | None = None,
        show_hist: bool | None = None,
        show_density: bool | None = None,
        norm: colors.Normalize | None = None,
        backend: Backend = "matplotlib",
        figsize: Tuple[float, float] = (8, 8),
        xlim: Tuple[float, float] | None = None,
        ylim: Tuple[float, float] | None = None,
        reg_method: str | bool = "ols",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        skill_table: Union[str, List[str], Mapping[str, str], bool] | None = None,
        ax: Axes | None = None,
        **kwargs,
    ) -> Axes | list[Axes]:
        """Scatter plot tailored for comparing model output with observations.

        Optionally, with density histogram.

        Parameters
        ----------
        bins: (int, float, sequence), optional
            bins for the 2D histogram on the background. By default 120 bins.
            if int, represents the number of bins of 2D
            if float, represents the bin size
            if sequence (list of int or float), represents the bin edges
        quantiles: (int, sequence), optional
            number of quantiles for QQ-plot, by default None and will depend
            on the scatter data length (10, 100 or 1000); if int, this is
            the number of points; if sequence (list of floats), represents
            the desired quantiles (from 0 to 1)
        fit_to_quantiles: bool, optional, by default False
            by default the regression line is fitted to all data, if True,
            it is fitted to the quantiles which can be useful to represent
            the extremes of the distribution, by default False
        show_points : (bool, int, float), optional
            Should the scatter points be displayed? None means: show all
            points if fewer than 1e4, otherwise show 1e4 sample points,
            by default None. float: fraction of points to show on plot
            from 0 to 1. e.g. 0.5 shows 50% of the points. int: if 'n' (int)
            given, then 'n' points will be displayed, randomly selected
        show_hist : bool, optional
            show the data density as a a 2d histogram, by default None
        show_density: bool, optional
            show the data density as a colormap of the scatter, by default
            None. If both `show_density` and `show_hist` are None, then
            `show_density` is used by default. If number of points is less
            than 200, then `show_density` is False as default.
            For binning the data, the kword `bins=Float` is used.
        norm : matplotlib.colors norm
            colormap normalization. If None, defaults to
            matplotlib.colors.PowerNorm(vmin=1, gamma=0.5)
        backend : str, optional
            use "plotly" (interactive) or "matplotlib" backend,
            by default "matplotlib"
        figsize : tuple, optional
            width and height of the figure, by default (8, 8)
        xlim : tuple, optional
            plot range for the observation (xmin, xmax), by default None
        ylim : tuple, optional
            plot range for the model (ymin, ymax), by default None
        reg_method : str or bool, optional
            method for determining the regression line
            "ols" : ordinary least squares regression
            "odr" : orthogonal distance regression,
            False : no regression line,
            by default "ols"
        title : str, optional
            plot title, by default None
        xlabel : str, optional
            x-label text on plot, by default None
        ylabel : str, optional
            y-label text on plot, by default None
        skill_table: str, List[str], dict[str,str], bool, optional
            list of modelskill.metrics or boolean, if True then by default modelskill.options.metrics.list.
            This kword adds a box at the right of the scatter plot.
            mapping can be used to rename the metrics in the table.
            by default False
        ax : matplotlib axes, optional
            axes to plot on, by default None
        **kwargs
            other keyword arguments to matplotlib.pyplot.scatter()

        Examples
        ------
        >>> cc.plot.scatter()
        >>> cc.plot.scatter(bins=0.2, backend='plotly')
        >>> cc.plot.scatter(show_points=False, title='no points')
        >>> cc.plot.scatter(xlabel='all observations', ylabel='my model')
        >>> cc.sel(model='HKZN_v2').plot.scatter(figsize=(10, 10))
        >>> cc.sel(observations=['c2','HKNA']).plot.scatter()
        """

        validate_backend(backend)
        reject_matplotlib_axes(ax, backend)

        cc = self.cc

        mod_names = cc.mod_names
        axes = []
        for mod_name in mod_names:
            ax_mod = self._scatter_one_model(
                mod_name=mod_name,
                bins=bins,
                quantiles=quantiles,
                fit_to_quantiles=fit_to_quantiles,
                show_points=show_points,
                show_hist=show_hist,
                show_density=show_density,
                norm=norm,
                backend=backend,
                figsize=figsize,
                xlim=xlim,
                ylim=ylim,
                reg_method=reg_method,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                skill_table=skill_table,
                ax=ax,
                **kwargs,
            )
            axes.append(ax_mod)
        return axes[0] if len(axes) == 1 else axes

    def _scatter_one_model(
        self,
        *,
        mod_name: str,
        bins: int | float,
        quantiles: int | Sequence[float] | None,
        fit_to_quantiles: bool,
        show_points: bool | int | float | None,
        show_hist: bool | None,
        show_density: bool | None,
        backend: Backend,
        figsize: Tuple[float, float],
        xlim: Tuple[float, float] | None,
        ylim: Tuple[float, float] | None,
        reg_method: str | bool,
        title: str | None,
        xlabel: str | None,
        ylabel: str | None,
        skill_table: Union[str, List[str], Mapping[str, str], bool] | None,
        ax,
        **kwargs,
    ):
        assert (
            mod_name in self.cc.mod_names
        ), f"Model {mod_name} not found in collection {self.cc.mod_names}"

        cc_sel_mod = self.cc.sel(model=mod_name)

        if cc_sel_mod.n_points == 0:
            raise ValueError("No data found in selection")

        df = cc_sel_mod._to_long_dataframe()
        x = df.obs_val.values
        y = df.mod_val.values

        # TODO why the first?
        unit_text = self.cc[0]._unit_text

        xlabel = xlabel or f"Observation, {unit_text}"
        ylabel = ylabel or f"Model, {unit_text}"
        title = title or f"{mod_name} vs {cc_sel_mod._name}"

        skill = None
        skill_score_unit = None

        if skill_table:
            metrics = None if skill_table is True else skill_table

            # TODO why is this here?
            if isinstance(self, ComparerCollectionPlotter) and len(cc_sel_mod) == 1:
                skill = cc_sel_mod.skill(metrics=metrics)  # type: ignore
            else:
                skill = cc_sel_mod.mean_skill(metrics=metrics)  # type: ignore
            # TODO improve this
            try:
                skill_score_unit = unit_text.split("[")[1].split("]")[0]
            except IndexError:
                skill_score_unit = ""  # Dimensionless

        if self.is_directional:
            # hide quantiles and regression line
            quantiles = 0
            reg_method = False

        skill_scores = skill.iloc[0].to_dict() if skill is not None else None

        ax = scatter(
            x=x,
            y=y,
            bins=bins,
            quantiles=quantiles,
            fit_to_quantiles=fit_to_quantiles,
            show_points=show_points,
            show_hist=show_hist,
            show_density=show_density,
            backend=backend,
            figsize=figsize,
            xlim=xlim,
            ylim=ylim,
            reg_method=reg_method,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            skill_scores=skill_scores,
            skill_score_unit=skill_score_unit,
            ax=ax,
            **kwargs,
        )

        if backend == "matplotlib" and self.is_directional:
            _xtick_directional(ax, xlim)
            _ytick_directional(ax, ylim)

        return ax

    def kde(
        self,
        *,
        ax=None,
        figsize=None,
        title=None,
        backend: Backend = "matplotlib",
        **kwargs,
    ):
        """Plot kernel density estimate of observation and model data.

        Parameters
        ----------
        ax : Axes, optional
            matplotlib axes (matplotlib backend only), by default None
        figsize : tuple, optional
            width and height of the figure in inches, by default None
        title : str, optional
            plot title, by default None
        backend : str, optional
            "matplotlib" (static) or "plotly" (interactive),
            by default "matplotlib"
        **kwargs
            passed to pandas.DataFrame.plot.kde() (matplotlib backend) or
            fig.update_layout() (plotly backend); `bw_method` is passed to
            the kernel density estimate by both backends

        Returns
        -------
        Axes or plotly.graph_objects.Figure

        Examples
        --------
        >>> cc.plot.kde()
        >>> cc.plot.kde(bw_method=0.5)
        >>> cc.plot.kde(bw_method='silverman')

        """
        validate_backend(backend)
        reject_matplotlib_axes(ax, backend)

        df = self.cc._to_long_dataframe()
        title = (
            _default_univarate_title("Density plot", self.cc)
            if title is None
            else title
        )

        if backend == "plotly":
            series = {"Observation": df.obs_val.values}
            series.update(
                {m: df[df.model == m].mod_val.values for m in self.cc.mod_names}
            )
            return _plotly.kde(
                series=series,
                title=title,
                xlabel=self.cc._unit_text,
                figsize=figsize,
                directional=self.is_directional,
                **kwargs,
            )

        _, ax = _get_fig_ax(ax, figsize)

        ax = df.obs_val.plot.kde(
            ax=ax, linestyle="dashed", label="Observation", **kwargs
        )

        for model in self.cc.mod_names:
            df_model = df[df.model == model]
            df_model.mod_val.plot.kde(ax=ax, label=model, **kwargs)

        ax.set_xlabel(f"{self.cc._unit_text}")

        ax.set_title(title)
        ax.legend()

        # remove y-axis, ticks and label
        ax.yaxis.set_visible(False)
        ax.tick_params(axis="y", which="both", length=0)
        ax.set_ylabel("")

        # remove box around plot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        if self.is_directional:
            _xtick_directional(ax)

        return ax

    def hist(
        self,
        bins: int | Sequence = 100,
        *,
        model: str | int | None = None,
        title: str | None = None,
        density: bool = True,
        alpha: float = 0.5,
        ax=None,
        figsize: Tuple[float, float] | None = None,
        backend: Backend = "matplotlib",
        **kwargs,
    ):
        """Plot histogram of specific model and all observations.

        Parameters
        ----------
        bins : int, optional
            number of bins, by default 100
        title : str, optional
            plot title, default: observation name
        density: bool, optional
            If True, draw and return a probability density, by default True
        alpha : float, optional
            alpha transparency fraction, by default 0.5
        ax : matplotlib axes, optional
            axes to plot on (matplotlib backend only), by default None
        figsize : tuple, optional
            width and height of the figure in inches, by default None
        backend : str, optional
            "matplotlib" (static) or "plotly" (interactive),
            by default "matplotlib"
        **kwargs
            other keyword arguments to df.hist() (matplotlib backend) or
            fig.update_layout() (plotly backend)

        Returns
        -------
        Axes or plotly.graph_objects.Figure
            one per model, or a list if the collection has multiple models

        Examples
        --------
        >>> cc.plot.hist()
        >>> cc.plot.hist(bins=100)

        See also
        --------
        pandas.Series.hist
        matplotlib.axes.Axes.hist
        """
        validate_backend(backend)
        reject_matplotlib_axes(ax, backend)

        figs = []
        for mod_name in self.cc.mod_names:
            figs.append(
                self._hist_one_model(
                    mod_name=mod_name,
                    bins=bins,
                    title=title,
                    density=density,
                    alpha=alpha,
                    ax=ax,
                    figsize=figsize,
                    backend=backend,
                    **kwargs,
                )
            )
        return figs[0] if len(figs) == 1 else figs

    def _hist_one_model(
        self,
        *,
        mod_name: str,
        bins: int | Sequence,
        title: str | None,
        density: bool,
        alpha: float,
        ax,
        figsize: Tuple[float, float] | None,
        backend: Backend = "matplotlib",
        **kwargs,
    ):
        from ._comparison import MOD_COLORS

        assert (
            mod_name in self.cc.mod_names
        ), f"Model {mod_name} not found in collection"
        mod_idx = _get_idx(mod_name, self.cc.mod_names)

        title = (
            _default_univarate_title("Histogram", self.cc) if title is None else title
        )

        df = self.cc._to_long_dataframe()
        obs_color = self.cc[0].data["Observation"].attrs["color"]
        xlabel = f"{self.cc[df.observation.iloc[0]]._unit_text}"

        if backend == "plotly":
            return _plotly.histogram(
                series={
                    mod_name: df[df.model == mod_name].mod_val.values,
                    "observations": df.obs_val.values,
                },
                colors=[MOD_COLORS[mod_idx], obs_color],
                bins=bins,
                density=density,
                alpha=alpha,
                title=title,
                xlabel=xlabel,
                figsize=figsize,
                directional=self.is_directional,
                **kwargs,
            )

        _, ax = _get_fig_ax(ax, figsize)

        kwargs["alpha"] = alpha
        kwargs["density"] = density
        df.mod_val.hist(bins=bins, color=MOD_COLORS[mod_idx], ax=ax, **kwargs)
        df.obs_val.hist(bins=bins, color=obs_color, ax=ax, **kwargs)

        ax.legend([mod_name, "observations"])
        ax.set_title(title)
        ax.set_xlabel(xlabel)

        if density:
            ax.set_ylabel("density")
        else:
            ax.set_ylabel("count")

        if self.is_directional:
            _xtick_directional(ax)

        return ax

    def taylor(
        self,
        *,
        normalize_std: bool = False,
        aggregate_observations: bool = True,
        figsize: Tuple[float, float] = (7, 7),
        marker: str = "o",
        marker_size: float = 6.0,
        title: str = "Taylor diagram",
    ) -> Figure | None:
        """Taylor diagram for model skill comparison.

        Taylor diagram showing model std and correlation to observation
        in a single-quadrant polar plot, with r=std and theta=arccos(cc).

        Parameters
        ----------
        normalize_std : bool, optional
            plot model std normalized with observation std, default False
        aggregate_observations : bool, optional
            should multiple observations be aggregated before plotting
            (or shown individually), default True
        figsize : tuple, optional
            width and height of the figure (should be square), by default (7, 7)
        marker : str, optional
            marker type e.g. "x", "*", by default "o"
        marker_size : float, optional
            size of the marker, by default 6
        title : str, optional
            title of the plot, by default "Taylor diagram"

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        ------
        >>> cc.plot.taylor()
        >>> cc.sel(observation="c2").plot.taylor()
        >>> cc.plot.taylor(start="2017-10-28", figsize=(5,5))

        Notes
        ----------
        Copin, Y. (2018). https://gist.github.com/ycopin/3342888, Yannick Copin <yannick.copin@laposte.net>
        """

        if (not aggregate_observations) and (not normalize_std):
            raise ValueError(
                "aggregate_observations=False is only possible if normalize_std=True!"
            )

        skill_func = self.cc.mean_skill if aggregate_observations else self.cc.skill
        sk = skill_func(
            metrics=[mtr._std_obs, mtr._std_mod, mtr.cc],  # type: ignore
        )
        if sk is None:
            # TODO when does this make sense?
            return

        # TODO reduce duplication of code in the ComparerPlotter/ComparerCollectionPlotter
        df = sk.to_dataframe()
        ref_std = 1.0 if normalize_std else df.iloc[0]["_std_obs"]

        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.map("_".join)

        df = df.rename(columns={"_std_obs": "obs_std", "_std_mod": "std"})
        pts = [
            TaylorPoint(
                name=r.Index,
                obs_std=r.obs_std,
                std=r.std,
                cc=r.cc,
                marker=marker,
                marker_size=marker_size,
            )
            for r in df.itertuples()
        ]

        return taylor_diagram(
            obs_std=ref_std,
            points=pts,
            figsize=figsize,
            normalize_std=normalize_std,
            title=title,
        )

    def box(
        self,
        *,
        ax=None,
        figsize=None,
        title=None,
        backend: Backend = "matplotlib",
        **kwargs,
    ):
        """Plot box plot of observations and model data.

        Parameters
        ----------
        ax : Axes, optional
            matplotlib axes (matplotlib backend only), by default None
        figsize : tuple, optional
            width and height of the figure in inches, by default None
        title : str, optional
            plot title, by default None
        backend : str, optional
            "matplotlib" (static) or "plotly" (interactive),
            by default "matplotlib"
        **kwargs
            passed to pandas.DataFrame.plot.box() (matplotlib backend) or
            fig.update_layout() (plotly backend)

        Returns
        -------
        Axes or plotly.graph_objects.Figure

        Examples
        --------
        >>> cc.plot.box()
        >>> cc.plot.box(showmeans=True)
        >>> cc.plot.box(ax=ax, title="Box plot")
        """
        validate_backend(backend)
        reject_matplotlib_axes(ax, backend)

        df = self.cc._to_long_dataframe()

        unique_obs_cols = ["time", "x", "y", "observation"]
        df = df.set_index(unique_obs_cols)
        unique_obs_values = df[~df.duplicated()].obs_val.values

        data = {"Observation": unique_obs_values}
        for model in df.model.unique():
            df_model = df[df.model == model]
            data[model] = df_model.mod_val.values

        title = (
            _default_univarate_title("Box plot", self.cc) if title is None else title
        )

        if backend == "plotly":
            return _plotly.box(
                series=data,
                title=title,
                ylabel=f"{self.cc._unit_text}",
                figsize=figsize,
                directional=self.is_directional,
                **kwargs,
            )

        _, ax = _get_fig_ax(ax, figsize)

        data = {k: pd.Series(v) for k, v in data.items()}
        df = pd.DataFrame(data)

        if "grid" not in kwargs:
            kwargs["grid"] = True

        ax = df.plot.box(ax=ax, **kwargs)

        ax.set_ylabel(f"{self.cc._unit_text}")
        ax.set_title(title)

        if self.is_directional:
            _ytick_directional(ax)

        return ax

    def qq(
        self,
        quantiles: int | Sequence[float] | None = None,
        *,
        title=None,
        ax=None,
        figsize=None,
        backend: Backend = "matplotlib",
        **kwargs,
    ):
        """Make quantile-quantile (q-q) plot of model data and observations.

        Primarily used to compare multiple models.

        Parameters
        ----------
        quantiles: (int, sequence), optional
            number of quantiles for QQ-plot, by default None and will depend on the scatter data length (10, 100 or 1000)
            if int, this is the number of points
            if sequence (list of floats), represents the desired quantiles (from 0 to 1)
        title : str, optional
            plot title, default: "Q-Q plot for [observation name]"
        ax : matplotlib.axes.Axes, optional
            axes to plot on (matplotlib backend only), by default None
        figsize : tuple, optional
            figure size in inches, by default None
        backend : str, optional
            "matplotlib" (static) or "plotly" (interactive),
            by default "matplotlib"
        **kwargs
            other keyword arguments to plt.plot() (matplotlib backend) or
            fig.update_layout() (plotly backend)

        Returns
        -------
        Axes or plotly.graph_objects.Figure

        Examples
        --------
        >>> cc.plot.qq()

        """
        validate_backend(backend)
        reject_matplotlib_axes(ax, backend)

        cc = self.cc
        df = cc._to_long_dataframe()
        title = (
            _default_univarate_title("Q-Q plot for ", self.cc)
            if title is None
            else title
        )

        if backend == "plotly":
            quantile_pairs = {}
            for model in cc.mod_names:
                df_model = df[df.model == model]
                quantile_pairs[model] = quantiles_xy(
                    df_model.obs_val.values, df_model.mod_val.values, quantiles
                )
            return _plotly.qq(
                quantiles=quantile_pairs,
                title=title,
                xlabel="Observation, " + cc._unit_text,
                ylabel="Model, " + cc._unit_text,
                figsize=figsize,
                directional=self.is_directional,
                **kwargs,
            )

        _, ax = _get_fig_ax(ax, figsize)

        xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf

        for model in cc.mod_names:
            df_model = df[df.model == model]

            x = df_model.obs_val.values
            y = df_model.mod_val.values
            xq, yq = quantiles_xy(x, y, quantiles)

            xmin = min([x.min(), xmin])
            xmax = max([x.max(), xmax])
            ymin = min([y.min(), ymin])
            ymax = max([y.max(), ymax])

            ax.plot(xq, yq, ".-", label=model, zorder=4, **kwargs)

        # 1:1 line
        ax.plot(
            [xmin, xmax],
            [ymin, ymax],
            label=options.plot.scatter.oneone_line.label,
            c=options.plot.scatter.oneone_line.color,
            zorder=3,
        )

        ax.axis("square")
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.minorticks_on()
        ax.grid(which="both", axis="both", linewidth="0.2", color="k", alpha=0.6)

        ax.legend()
        ax.set_xlabel("Observation, " + cc._unit_text)
        ax.set_ylabel("Model, " + cc._unit_text)
        ax.set_title(title)

        if self.is_directional:
            _xtick_directional(ax)
            _ytick_directional(ax)

        return ax

    def residual_hist(
        self,
        bins=100,
        title=None,
        color=None,
        figsize=None,
        ax=None,
        backend: Backend = "matplotlib",
        **kwargs,
    ):
        """plot histogram of residual values

        Parameters
        ----------
        bins : int, optional
            specification of bins, by default 100
        title : str, optional
            plot title, default: Residuals, [name]
        color : str, optional
            residual color, by default "#8B8D8E"
        figsize : tuple, optional
            figure size in inches, by default None
        ax : Axes | list[Axes], optional
            axes to plot on (matplotlib backend only), by default None
        backend : str, optional
            "matplotlib" (static) or "plotly" (interactive),
            by default "matplotlib"
        **kwargs
            other keyword arguments to plt.hist() (matplotlib backend) or
            fig.update_layout() (plotly backend)

        Returns
        -------
        Axes or plotly.graph_objects.Figure
            one per model, or a list if the collection has multiple models
        """
        validate_backend(backend)
        reject_matplotlib_axes(ax, backend)

        cc = self.cc

        if cc.n_models == 1:
            return self._residual_hist_one_model(
                bins=bins,
                title=title,
                color=color,
                figsize=figsize,
                ax=ax,
                mod_name=cc.mod_names[0],
                backend=backend,
                **kwargs,
            )

        if ax is not None and len(ax) != len(cc.mod_names):
            raise ValueError("Number of axes must match number of models")

        axs = ax if ax is not None else [None] * len(cc.mod_names)

        for i, mod_name in enumerate(cc.mod_names):
            cc_model = cc.sel(model=mod_name)
            ax_mod = cc_model.plot.residual_hist(
                bins=bins,
                title=title,
                color=color,
                figsize=figsize,
                ax=axs[i],
                backend=backend,
                **kwargs,
            )
            axs[i] = ax_mod

        return axs  # type: ignore[return-value]  # axs starts as list[None] and is populated in the loop; mypy cannot infer that all slots are filled with Axes

    def _residual_hist_one_model(
        self,
        bins=100,
        title=None,
        color=None,
        figsize=None,
        ax=None,
        mod_name=None,
        backend: Backend = "matplotlib",
        **kwargs,
    ):
        """Residual histogram for one model only"""
        df = self.cc.sel(model=mod_name)._to_long_dataframe()
        residuals = df.mod_val.values - df.obs_val.values

        title = (
            _default_univarate_title(f"Residuals, Model {mod_name}", self.cc)
            if title is None
            else title
        )
        xlabel = f"Residuals of {self.cc._unit_text}"

        if backend == "plotly":
            return _plotly.residual_hist(
                residuals=residuals,
                bins=bins,
                color=color,
                title=title,
                xlabel=xlabel,
                figsize=figsize,
                directional=self.is_directional,
                **kwargs,
            )

        _, ax = _get_fig_ax(ax, figsize)

        color = _plotly.RESIDUAL_COLOR if color is None else color
        ax.hist(residuals, bins=bins, color=color, **kwargs)
        ax.set_title(title)
        ax.set_xlabel(xlabel)

        if self.is_directional:
            ticks = np.linspace(-180, 180, 9)
            ax.set_xticks(ticks)
            ax.set_xlim(-180, 180)

        return ax

    def spatial_overview(
        self,
        ax=None,
        figsize: Tuple | None = None,
        title: str | None = None,
    ) -> Axes:
        """Plot observation points on a map showing the model domain

        Parameters
        ----------
        ax: matplotlib.axes, optional
            Adding to existing axis, instead of creating new fig
        figsize : (float, float), optional
            figure size, by default None
        title: str, optional
            plot title, default empty

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes object
        """
        from ..plotting import spatial_overview

        obs = [cmp._to_observation() for cmp in self.cc]
        # TODO how to add model domain(s)

        return spatial_overview(obs, ax=ax, figsize=figsize, title=title)

    def temporal_coverage(
        self,
        limit_to_model_period: bool = True,
        marker: str = "_",
        ax: Any | None = None,
        figsize: Any | None = None,
        title: Any | None = None,
    ) -> Axes:
        """Plot graph showing temporal coverage for all observations and models

        Parameters
        ----------
        limit_to_model_period : bool, optional
            Show temporal coverage only for period covered
            by the model, by default True
        marker : str, optional
            plot marker for observations, by default "_"
        ax: matplotlib.axes, optional
            Adding to existing axis, instead of creating new fig
        figsize : Tuple(float, float), optional
            size of figure, by default (7, 0.45*n_lines)
        title: str, optional
            plot title, default empty
        """
        from ..plotting import temporal_coverage

        obs = [cmp._to_observation() for cmp in self.cc]
        mod = self.cc[0]._to_model()

        return temporal_coverage(
            obs=obs,
            mod=mod,
            limit_to_model_period=limit_to_model_period,
            marker=marker,
            ax=ax,
            figsize=figsize,
            title=title,
        )
