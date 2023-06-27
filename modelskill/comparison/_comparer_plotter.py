from typing import Union, List, Optional, Tuple

import matplotlib.pyplot as plt  # type: ignore

from .. import metrics as mtr
from ._utils import _get_id
from ..plot import colors, scatter, taylor_diagram, TaylorPoint


class ComparerPlotter:
    """Plotter class for Comparer"""
    def __init__(self, comparer):
        self.comparer = comparer

    def __call__(self, *args, **kwargs):
        """Plot scatter plot of modelled vs observed data"""
        return self.scatter(*args, **kwargs)

    def timeseries(
        self, title=None, *, ylim=None, figsize=None, backend="matplotlib", **kwargs
    ):
        """Timeseries plot showing compared data: observation vs modelled

        Parameters
        ----------
        title : str, optional
            plot title, by default None
        ylim : tuple, optional
            plot range for the model (ymin, ymax), by default None
        figsize : (float, float), optional
            figure size, by default None
        backend : str, optional
            use "plotly" (interactive) or "matplotlib" backend, by default "matplotlib"backend:

        Returns
        -------
        matplotlib.axes.Axes or plotly.graph_objects.Figure
        """
        from ._comparison import MOD_COLORS

        cmp = self.comparer

        if title is None:
            title = cmp.name

        if backend == "matplotlib":
            _, ax = plt.subplots(figsize=figsize)
            for j in range(cmp.n_models):
                key = cmp.mod_names[j]
                mod_df = cmp.raw_mod_data[key]
                mod_df[key].plot(ax=ax, color=MOD_COLORS[j])

            ax.scatter(
                cmp.time,
                cmp.data[cmp.obs_name].values,
                marker=".",
                color=cmp.data[cmp.obs_name].attrs["color"],
            )
            ax.set_ylabel(cmp.unit_text)
            ax.legend([*cmp.mod_names, cmp.obs_name])
            ax.set_ylim(ylim)
            plt.title(title)
            return ax

        elif backend == "plotly":  # pragma: no cover
            import plotly.graph_objects as go  # type: ignore

            mod_scatter_list = []
            for j in range(cmp.n_models):
                key = cmp.mod_names[j]
                mod_df = cmp.raw_mod_data[key]
                mod_scatter_list.append(
                    go.Scatter(
                        x=mod_df.index,
                        y=mod_df[key],
                        name=key,
                        line=dict(color=MOD_COLORS[j]),
                    )
                )

            fig = go.Figure(
                [
                    *mod_scatter_list,
                    go.Scatter(
                        x=cmp.time,
                        y=cmp.data[cmp.obs_name].values,
                        name=cmp.obs_name,
                        mode="markers",
                        marker=dict(color=cmp.data[cmp.obs_name].attrs["color"]),
                    ),
                ]
            )

            fig.update_layout(title=title, yaxis_title=cmp.unit_text, **kwargs)
            fig.update_yaxes(range=ylim)

            fig.show()
        else:
            raise ValueError(f"Plotting backend: {backend} not supported")

    def hist(
        self, *, model=None, bins=100, title=None, density=True, alpha=0.5, **kwargs
    ):
        """Plot histogram of model data and observations.

        Wraps pandas.DataFrame hist() method.

        Parameters
        ----------
        model : (str, int), optional
            name or id of model to be plotted, by default 0
        bins : int, optional
            number of bins, by default 100
        title : str, optional
            plot title, default: [model name] vs [observation name]
        density: bool, optional
            If True, draw and return a probability density
        alpha : float, optional
            alpha transparency fraction, by default 0.5
        kwargs : other keyword arguments to df.plot.hist()

        Returns
        -------
        matplotlib axes

        See also
        --------
        pandas.Series.plot.hist
        matplotlib.axes.Axes.hist
        """
        from ._comparison import MOD_COLORS  # TODO move to here

        cmp = self.comparer
        mod_id = _get_id(model, cmp.mod_names)
        mod_name = cmp.mod_names[mod_id]

        title = f"{mod_name} vs {cmp.name}" if title is None else title

        kwargs["alpha"] = alpha
        kwargs["density"] = density

        ax = (
            cmp.data[mod_name]
            .to_series()
            .hist(bins=bins, color=MOD_COLORS[mod_id], **kwargs)
        )

        cmp.data[cmp.obs_name].to_series().hist(
            bins=bins, color=cmp.data[cmp.obs_name].attrs["color"], ax=ax, **kwargs
        )
        ax.legend([mod_name, cmp.obs_name])
        plt.title(title)
        plt.xlabel(f"{cmp.unit_text}")
        if density:
            plt.ylabel("density")
        else:
            plt.ylabel("count")

        return ax

    def kde(self, ax=None, **kwargs):
        """Plot kde (kernel density estimates of distributions) of model data and observations.

        Wraps pandas.DataFrame kde() method.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            axes to plot on, by default None
        kwargs : other keyword arguments to df.plot.kde()

        Returns
        -------
        matplotlib axes

        See also
        --------
        pandas.Series.plot.kde
        """
        cmp = self.comparer

        if ax is None:
            ax = plt.gca()

        cmp.data.Observation.to_series().plot.kde(
            ax=ax, linestyle="dashed", label="Observation", **kwargs
        )

        for model in cmp.mod_names:
            cmp.data[model].to_series().plot.kde(ax=ax, label=model, **kwargs)

        ax.set_xlabel(cmp.unit_text)  # TODO

        ax.legend()

        # remove y-axis
        ax.yaxis.set_visible(False)
        # remove y-ticks
        ax.tick_params(axis="y", which="both", length=0)
        # remove y-label
        ax.set_ylabel("")

        # remove box around plot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        return ax

    def scatter(
        self,
        *,
        model=None,
        bins: Union[int, float, List[int], List[float]] = 20,
        quantiles: Optional[Union[int, List[float]]] = None,
        fit_to_quantiles: bool = False,
        show_points: Optional[Union[bool, int, float]] = None,
        show_hist: Optional[bool] = None,
        show_density: Optional[bool] = None,
        norm: Optional[colors.Normalize] = None,
        backend: str = "matplotlib",
        figsize: Tuple[float, float] = (8, 8),
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        reg_method: str = "ols",
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        skill_table: Optional[Union[str, List[str], bool]] = None,
        **kwargs,
    ):
        """Scatter plot showing compared data: observation vs modelled
        Optionally, with density histogram.

        Parameters
        ----------
        model : (str, int), optional
            name or id of model to be plotted, by default 0
        bins: (int, float, sequence), optional
            bins for the 2D histogram on the background. By default 20 bins.
            if int, represents the number of bins of 2D
            if float, represents the bin size
            if sequence (list of int or float), represents the bin edges
        quantiles: (int, sequence), optional
            number of quantiles for QQ-plot, by default None and will depend on the scatter data length (10, 100 or 1000)
            if int, this is the number of points
            if sequence (list of floats), represents the desired quantiles (from 0 to 1)
        fit_to_quantiles: bool, optional, by default False
            by default the regression line is fitted to all data, if True, it is fitted to the quantiles
            which can be useful to represent the extremes of the distribution
        show_points : (bool, int, float), optional
            Should the scatter points be displayed?
            None means: show all points if fewer than 1e4, otherwise show 1e4 sample points, by default None.
            float: fraction of points to show on plot from 0 to 1. eg 0.5 shows 50% of the points.
            int: if 'n' (int) given, then 'n' points will be displayed, randomly selected
        show_hist : bool, optional
            show the data density as a a 2d histogram, by default None
        show_density: bool, optional
            show the data density as a colormap of the scatter, by default None. If both `show_density` and `show_hist`
        norm : matplotlib.colors norm
            colormap normalization
            If None, defaults to matplotlib.colors.PowerNorm(vmin=1,gamma=0.5)
        are None, then `show_density` is used by default.
            for binning the data, the previous kword `bins=Float` is used
        backend : str, optional
            use "plotly" (interactive) or "matplotlib" backend, by default "matplotlib"
        figsize : tuple, optional
            width and height of the figure, by default (8, 8)
        xlim : tuple, optional
            plot range for the observation (xmin, xmax), by default None
        ylim : tuple, optional
            plot range for the model (ymin, ymax), by default None
        reg_method : str, optional
            method for determining the regression line
            "ols" : ordinary least squares regression
            "odr" : orthogonal distance regression,
            by default "ols"
        title : str, optional
            plot title, by default None
        xlabel : str, optional
            x-label text on plot, by default None
        ylabel : str, optional
            y-label text on plot, by default None
        skill_table : str, List[str], bool, optional
            list of modelskill.metrics or boolean, if True then by default modelskill.options.metrics.list.
            This kword adds a box at the right of the scatter plot,
            by default False
        kwargs

        Examples
        ------
        >>> cmp.plot.scatter()
        >>> cmp.plot.scatter(bins=0.2, backend='plotly')
        >>> cmp.plot.scatter(show_points=False, title='no points')
        >>> cm.plot.scatter(xlabel='all observations', ylabel='my model')
        >>> cmp.sel(model='HKZN_v2').plot.scatter(figsize=(10, 10))
        """

        cmp = self.comparer

        cmp = self.comparer
        mod_id = _get_id(model, cmp.mod_names)
        mod_name = cmp.mod_names[mod_id]

        if cmp.n_points == 0:
            raise ValueError("No data found in selection")

        x = cmp.data.Observation.values
        y = cmp.data[mod_name].values

        assert x.ndim == y.ndim == 1, "x and y must be 1D arrays"
        assert x.shape == y.shape, "x and y must have the same shape"

        unit_text = cmp.unit_text
        xlabel = xlabel or f"Observation, {unit_text}"
        ylabel = ylabel or f"Model, {unit_text}"
        title = title or f"{mod_name} vs {cmp.name}"

        skill_df = None
        units = None

        if skill_table:
            metrics = None if skill_table is True else skill_table
            skill_df = cmp.skill(metrics=metrics)
            try:
                units = unit_text.split("[")[1].split("]")[0]
            except IndexError:
                units = ""  # Dimensionless

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
            skill_df=skill_df,
            units=units,
            **kwargs,
        )
        return ax

    def taylor(
        self,
        normalize_std: bool = False,
        figsize: Tuple[float, float] = (7, 7),
        marker: str = "o",
        marker_size: float = 6.0,
        title: str = "Taylor diagram",
    ) -> None:
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
        metrics = [
            mtr._std_obs,
            mtr._std_mod,
            mtr.cc,
        ]

        s = cmp.skill(metrics=metrics)

        if s is None:  # TODO
            return
        df = s.df
        ref_std = 1.0 if normalize_std else df.iloc[0]["_std_obs"]

        df = df[["_std_obs", "_std_mod", "cc"]].copy()
        df.columns = ["obs_std", "std", "cc"]

        pts = [
            TaylorPoint(
                r.Index, r.obs_std, r.std, r.cc, marker=marker, marker_size=marker_size
            )
            for r in df.itertuples()
        ]

        taylor_diagram(
            obs_std=ref_std,
            points=pts,
            figsize=figsize,
            obs_text=f"Obs: {cmp.name}",
            normalize_std=normalize_std,
            title=title,
        )

    def residual_hist(self, bins=100, title=None, color=None, **kwargs):
        """plot histogram of residual values

        Parameters
        ----------
        bins : int, optional
            specification of bins, by default 100
        title : str, optional
            plot title, default: Residuals, [name]
        color : str, optional
            residual color, by default "#8B8D8E"
        kwargs : other keyword arguments to plt.hist()
        """

        default_color = "#8B8D8E"
        color = default_color if color is None else color
        title = f"Residuals, {self.comparer.name}" if title is None else title
        plt.hist(self.comparer.residual, bins=bins, color=color, **kwargs)
        plt.title(title)
        plt.xlabel(f"Residuals of {self.comparer.unit_text}")
