from __future__ import annotations
from typing import (
    Literal,
    Union,
    List,
    Optional,
    Tuple,
    Sequence,
    TYPE_CHECKING,
    Callable,
)
import warnings

if TYPE_CHECKING:
    import matplotlib.figure
    import matplotlib.axes
    from ._comparison import Comparer

import numpy as np  # type: ignore

from .. import metrics as mtr
from ..utils import _get_idx
import matplotlib.colors as colors
from ..plotting._misc import (
    _get_fig_ax,
    _xtick_directional,
    _ytick_directional,
    quantiles_xy,
)
from ..plotting import taylor_diagram, scatter, TaylorPoint
from ..settings import options


class ComparerPlotter:
    """Plotter class for Comparer

    Examples
    --------
    >>> cmp.plot.scatter()
    >>> cmp.plot.timeseries()
    >>> cmp.plot.hist()
    >>> cmp.plot.kde()
    >>> cmp.plot.qq()
    >>> cmp.plot.box()
    """

    def __init__(self, comparer: Comparer) -> None:
        self.comparer = comparer
        self.is_directional = comparer.quantity.is_directional

    def __call__(self, *args, **kwargs) -> matplotlib.axes.Axes:
        """Plot scatter plot of modelled vs observed data"""
        return self.scatter(*args, **kwargs)

    def timeseries(
        self,
        *,
        title: str | None = None,
        ylim: Tuple[float, float] | None = None,
        ax=None,
        figsize: Tuple[float, float] | None = None,
        backend: str = "matplotlib",
        **kwargs,
    ):
        """Timeseries plot showing compared data: observation vs modelled

        Parameters
        ----------
        title : str, optional
            plot title, by default None
        ylim : (float, float), optional
            plot range for the model (ymin, ymax), by default None
        ax : matplotlib.axes.Axes, optional
            axes to plot on, by default None
        figsize : (float, float), optional
            figure size, by default None
        backend : str, optional
            use "plotly" (interactive) or "matplotlib" backend,
            by default "matplotlib"
        **kwargs
            other keyword arguments to fig.update_layout (plotly backend)

        Returns
        -------
        matplotlib.axes.Axes or plotly.graph_objects.Figure
        """
        from ._comparison import MOD_COLORS

        cmp = self.comparer

        if title is None:
            title = cmp.name

        if backend == "matplotlib":
            fig, ax = _get_fig_ax(ax, figsize)
            for j in range(cmp.n_models):
                key = cmp.mod_names[j]
                mod = cmp.raw_mod_data[key]._values_as_series
                mod.plot(ax=ax, color=MOD_COLORS[j])

            ax.scatter(
                cmp.time,
                cmp.data[cmp._obs_name].values,
                marker=".",
                color=cmp.data[cmp._obs_name].attrs["color"],
            )
            ax.set_ylabel(cmp._unit_text)
            ax.legend([*cmp.mod_names, cmp._obs_name])
            ax.set_ylim(ylim)
            if self.is_directional:
                _ytick_directional(ax, ylim)
            ax.set_title(title)
            return ax

        elif backend == "plotly":  # pragma: no cover
            import plotly.graph_objects as go  # type: ignore

            mod_scatter_list = []
            for j in range(cmp.n_models):
                key = cmp.mod_names[j]
                mod = cmp.raw_mod_data[key]._values_as_series
                mod_scatter_list.append(
                    go.Scatter(
                        x=mod.index,
                        y=mod.values,
                        name=key,
                        line=dict(color=MOD_COLORS[j]),
                    )
                )

            fig = go.Figure(
                [
                    *mod_scatter_list,
                    go.Scatter(
                        x=cmp.time,
                        y=cmp.data[cmp._obs_name].values,
                        name=cmp._obs_name,
                        mode="markers",
                        marker=dict(color=cmp.data[cmp._obs_name].attrs["color"]),
                    ),
                ]
            )

            fig.update_layout(title=title, yaxis_title=cmp._unit_text, **kwargs)
            fig.update_yaxes(range=ylim)

            return fig
        else:
            raise ValueError(f"Plotting backend: {backend} not supported")

    def hist(
        self,
        bins: int | Sequence = 100,
        *,
        model: str | int | None = None,
        title: str | None = None,
        ax=None,
        figsize: Tuple[float, float] | None = None,
        density: bool = True,
        alpha: float = 0.5,
        **kwargs,
    ):
        """Plot histogram of model data and observations.

        Wraps pandas.DataFrame hist() method.

        Parameters
        ----------
        bins : int, optional
            number of bins, by default 100
        title : str, optional
            plot title, default: [model name] vs [observation name]
        ax : matplotlib.axes.Axes, optional
            axes to plot on, by default None
        figsize : tuple, optional
            figure size, by default None
        density: bool, optional
            If True, draw and return a probability density
        alpha : float, optional
            alpha transparency fraction, by default 0.5
        **kwargs
            other keyword arguments to df.plot.hist()

        Returns
        -------
        matplotlib axes

        See also
        --------
        pandas.Series.plot.hist
        matplotlib.axes.Axes.hist
        """
        cmp = self.comparer

        if model is None:
            mod_names = cmp.mod_names
        else:
            warnings.warn(
                "The 'model' keyword is deprecated! Instead, filter comparer before plotting cmp.sel(model=...).plot.hist()",
                FutureWarning,
            )
            model_list = [model] if isinstance(model, (str, int)) else model
            mod_names = [cmp.mod_names[_get_idx(m, cmp.mod_names)] for m in model_list]

        axes = []
        for mod_name in mod_names:
            ax_mod = self._hist_one_model(
                mod_name=mod_name,
                bins=bins,
                title=title,
                ax=ax,
                figsize=figsize,
                density=density,
                alpha=alpha,
                **kwargs,
            )
            axes.append(ax_mod)

        return axes[0] if len(axes) == 1 else axes

    def _hist_one_model(
        self,
        *,
        mod_name: str,
        bins: int | Sequence | None,
        title: str | None,
        ax,
        figsize: Tuple[float, float] | None,
        density: bool | None,
        alpha: float | None,
        **kwargs,
    ):
        from ._comparison import MOD_COLORS  # TODO move to here

        cmp = self.comparer
        assert mod_name in cmp.mod_names, f"Model {mod_name} not found in comparer"
        mod_idx = _get_idx(mod_name, cmp.mod_names)

        title = f"{mod_name} vs {cmp.name}" if title is None else title

        _, ax = _get_fig_ax(ax, figsize)

        kwargs["alpha"] = alpha
        kwargs["density"] = density
        kwargs["ax"] = ax

        ax = (
            cmp.data[mod_name]
            .to_series()
            .hist(bins=bins, color=MOD_COLORS[mod_idx], **kwargs)
        )

        cmp.data[cmp._obs_name].to_series().hist(
            bins=bins, color=cmp.data[cmp._obs_name].attrs["color"], **kwargs
        )
        ax.legend([mod_name, cmp._obs_name])
        ax.set_title(title)
        ax.set_xlabel(f"{cmp._unit_text}")
        if density:
            ax.set_ylabel("density")
        else:
            ax.set_ylabel("count")

        if self.is_directional:
            _xtick_directional(ax)

        return ax

    def kde(self, ax=None, title=None, figsize=None, **kwargs) -> matplotlib.axes.Axes:
        """Plot kde (kernel density estimates of distributions) of model data and observations.

        Wraps pandas.DataFrame kde() method.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            axes to plot on, by default None
        title : str, optional
            plot title, default: "KDE plot for [observation name]"
        figsize : tuple, optional
            figure size, by default None
        **kwargs
            other keyword arguments to df.plot.kde()

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> cmp.plot.kde()
        >>> cmp.plot.kde(bw_method=0.3)
        >>> cmp.plot.kde(ax=ax, bw_method='silverman')
        >>> cmp.plot.kde(xlim=[0,None], title="Density plot");

        See also
        --------
        pandas.Series.plot.kde
        """
        cmp = self.comparer

        _, ax = _get_fig_ax(ax, figsize)

        cmp.data.Observation.to_series().plot.kde(
            ax=ax, linestyle="dashed", label="Observation", **kwargs
        )

        for model in cmp.mod_names:
            cmp.data[model].to_series().plot.kde(ax=ax, label=model, **kwargs)

        ax.set_xlabel(cmp._unit_text)  # TODO

        ax.legend()

        # remove y-axis, ticks and label
        ax.yaxis.set_visible(False)
        ax.tick_params(axis="y", which="both", length=0)
        ax.set_ylabel("")
        title = f"KDE plot for {cmp.name}" if title is None else title
        ax.set_title(title)

        # remove box around plot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        if self.is_directional:
            _xtick_directional(ax)

        return ax

    def qq(
        self,
        quantiles: int | Sequence[float] | None = None,
        *,
        title=None,
        ax=None,
        figsize=None,
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
            axes to plot on, by default None
        figsize : tuple, optional
            figure size, by default None
        **kwargs
            other keyword arguments to plt.plot()

        Returns
        -------
        matplotlib axes

        Examples
        --------
        >>> cmp.plot.qq()

        """
        cmp = self.comparer

        _, ax = _get_fig_ax(ax, figsize)

        x = cmp.data.Observation.values
        xmin, xmax = x.min(), x.max()
        ymin, ymax = np.inf, -np.inf

        for mod_name in cmp.mod_names:
            y = cmp.data[mod_name].values
            ymin = min([y.min(), ymin])
            ymax = max([y.max(), ymax])
            xq, yq = quantiles_xy(x, y, quantiles)
            ax.plot(
                xq,
                yq,
                ".-",
                label=mod_name,
                zorder=4,
                **kwargs,
            )

        xymin = min([xmin, ymin])
        xymax = max([xmax, ymax])

        # 1:1 line
        ax.plot(
            [xymin, xymax],
            [xymin, xymax],
            label=options.plot.scatter.oneone_line.label,
            c=options.plot.scatter.oneone_line.color,
            zorder=3,
        )

        ax.axis("square")
        ax.set_xlim([xymin, xymax])
        ax.set_ylim([xymin, xymax])
        ax.minorticks_on()
        ax.grid(which="both", axis="both", linewidth="0.2", color="k", alpha=0.6)

        ax.legend()
        ax.set_xlabel("Observation, " + cmp._unit_text)
        ax.set_ylabel("Model, " + cmp._unit_text)
        ax.set_title(title or f"Q-Q plot for {cmp.name}")

        if self.is_directional:
            _xtick_directional(ax)
            _ytick_directional(ax)

        return ax

    def box(self, *, ax=None, title=None, figsize=None, **kwargs):
        """Make a box plot of model data and observations.

        Wraps pandas.DataFrame boxplot() method.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            axes to plot on, by default None
        title : str, optional
            plot title, default: [observation name]
        figsize : tuple, optional
            figure size, by default None
        **kwargs
            other keyword arguments to df.boxplot()

        Returns
        -------
        matplotlib axes

        Examples
        --------
        >>> cmp.plot.box()
        >>> cmp.plot.box(showmeans=True)
        >>> cmp.plot.box(ax=ax, title="Box plot")

        See also
        --------
        pandas.DataFrame.boxplot
        matplotlib.pyplot.boxplot
        """
        cmp = self.comparer

        _, ax = _get_fig_ax(ax, figsize)

        cols = ["Observation"] + cmp.mod_names
        df = cmp.data[cols].to_dataframe()[cols]
        df.boxplot(ax=ax, **kwargs)
        ax.set_ylabel(cmp._unit_text)
        ax.set_title(title or cmp.name)

        if self.is_directional:
            _ytick_directional(ax)

        return ax

    def scatter(
        self,
        *,
        model=None,
        bins: int | float = 120,
        quantiles: int | Sequence[float] | None = None,
        fit_to_quantiles: bool = False,
        show_points: bool | int | float | None = None,
        show_hist: Optional[bool] = None,
        show_density: Optional[bool] = None,
        norm: Optional[colors.Normalize] = None,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: Tuple[float, float] = (8, 8),
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        reg_method: str | bool = "ols",
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        skill_table: Optional[Union[str, List[str], bool]] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Scatter plot showing compared data: observation vs modelled
        Optionally, with density histogram.

        Parameters
        ----------
        bins: (int, float, sequence), optional
            bins for the 2D histogram on the background. By default 20 bins.
            if int, represents the number of bins of 2D
            if float, represents the bin size
            if sequence (list of int or float), represents the bin edges
        quantiles: (int, sequence), optional
            number of quantiles for QQ-plot, by default None and will depend
            on the scatter data length (10, 100 or 1000); if int, this is
            the number of points; if sequence (list of floats), represents
            the desired quantiles (from 0 to 1)
        fit_to_quantiles: bool, optional
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
            `show_density` is used by default. For binning the data, the
            kword `bins=Float` is used.
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
            False : no regression line
            by default "ols"
        title : str, optional
            plot title, by default None
        xlabel : str, optional
            x-label text on plot, by default None
        ylabel : str, optional
            y-label text on plot, by default None
        skill_table : str, List[str], bool, optional
            list of modelskill.metrics or boolean, if True then by default
            modelskill.options.metrics.list. This kword adds a box at the
            right of the scatter plot, by default False
        ax : matplotlib.axes.Axes, optional
            axes to plot on, by default None
        **kwargs
            other keyword arguments to plt.scatter()

        Examples
        ------
        >>> cmp.plot.scatter()
        >>> cmp.plot.scatter(bins=0.2, backend='plotly')
        >>> cmp.plot.scatter(show_points=False, title='no points')
        >>> cmp.plot.scatter(xlabel='all observations', ylabel='my model')
        >>> cmp.sel(model='HKZN_v2').plot.scatter(figsize=(10, 10))
        """

        cmp = self.comparer
        if model is None:
            mod_names = cmp.mod_names
        else:
            warnings.warn(
                "The 'model' keyword is deprecated! Instead, filter comparer before plotting cmp.sel(model=...).plot.scatter()",
                FutureWarning,
            )
            model_list = [model] if isinstance(model, (str, int)) else model
            mod_names = [cmp.mod_names[_get_idx(m, cmp.mod_names)] for m in model_list]

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
        show_hist: Optional[bool],
        show_density: Optional[bool],
        norm: Optional[colors.Normalize],
        backend: Literal["matplotlib", "plotly"],
        figsize: Tuple[float, float],
        xlim: Optional[Tuple[float, float]],
        ylim: Optional[Tuple[float, float]],
        reg_method: str | bool,
        title: Optional[str],
        xlabel: Optional[str],
        ylabel: Optional[str],
        skill_table: Optional[Union[str, List[str], bool]],
        **kwargs,
    ):
        """Scatter plot for one model only"""

        cmp = self.comparer
        cmp_sel_mod = cmp.sel(model=mod_name)
        assert mod_name in cmp.mod_names, f"Model {mod_name} not found in comparer"

        if cmp_sel_mod.n_points == 0:
            raise ValueError("No data found in selection")

        x = cmp_sel_mod.data.Observation.values
        y = cmp_sel_mod.data[mod_name].values

        assert x.ndim == y.ndim == 1, "x and y must be 1D arrays"
        assert x.shape == y.shape, "x and y must have the same shape"

        unit_text = cmp._unit_text
        xlabel = xlabel or f"Observation, {unit_text}"
        ylabel = ylabel or f"Model, {unit_text}"
        title = title or f"{mod_name} vs {cmp.name}"

        skill = None
        skill_score_unit = None

        if skill_table:
            metrics = None if skill_table is True else skill_table
            skill = cmp_sel_mod.skill(metrics=metrics)  # type: ignore
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
            norm=norm,
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
            **kwargs,
        )

        if backend == "matplotlib" and self.is_directional:
            _xtick_directional(ax, xlim)
            _ytick_directional(ax, ylim)

        return ax

    def taylor(
        self,
        *,
        normalize_std: bool = False,
        figsize: Tuple[float, float] = (7, 7),
        marker: str = "o",
        marker_size: float = 6.0,
        title: str = "Taylor diagram",
    ):
        """Taylor diagram showing model std and correlation to observation
        in a single-quadrant polar plot, with r=std and theta=arccos(cc).

        Parameters
        ----------
        normalize_std : bool, optional
            plot model std normalized with observation std, default False
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
        >>> comparer.taylor()
        >>> comparer.taylor(start="2017-10-28", figsize=(5,5))

        References
        ----------
        Copin, Y. (2018). https://gist.github.com/ycopin/3342888, Yannick Copin <yannick.copin@laposte.net>
        """
        cmp = self.comparer

        # TODO consider if this round-trip  via mtr is necessary to get the std:s
        metrics: List[Callable] = [
            mtr._std_obs,
            mtr._std_mod,
            mtr.cc,
        ]

        sk = cmp.skill(metrics=metrics)

        if sk is None:  # TODO
            return
        df = sk.to_dataframe()
        ref_std = 1.0 if normalize_std else df.iloc[0]["_std_obs"]

        df = df[["_std_obs", "_std_mod", "cc"]].copy()
        df.columns = ["obs_std", "std", "cc"]

        pts = [
            TaylorPoint(
                r.Index, r.obs_std, r.std, r.cc, marker=marker, marker_size=marker_size
            )
            for r in df.itertuples()
        ]

        return taylor_diagram(
            obs_std=ref_std,
            points=pts,
            figsize=figsize,
            obs_text=f"Obs: {cmp.name}",
            normalize_std=normalize_std,
            title=title,
        )

    def residual_hist(
        self, bins=100, title=None, color=None, figsize=None, ax=None, **kwargs
    ) -> matplotlib.axes.Axes:
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
            figure size, by default None
        ax : matplotlib.axes.Axes, optional
            axes to plot on, by default None
        **kwargs
            other keyword arguments to plt.hist()

        Returns
        -------
        matplotlib.axes.Axes
        """
        _, ax = _get_fig_ax(ax, figsize)

        default_color = "#8B8D8E"
        color = default_color if color is None else color
        title = f"Residuals, {self.comparer.name}" if title is None else title
        ax.hist(self.comparer._residual, bins=bins, color=color, **kwargs)
        ax.set_title(title)
        ax.set_xlabel(f"Residuals of {self.comparer._unit_text}")

        if self.is_directional:
            ticks = np.linspace(-180, 180, 9)
            ax.set_xticks(ticks)
            ax.set_xlim(-180, 180)

        return ax
