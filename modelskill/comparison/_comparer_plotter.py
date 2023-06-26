from typing import Union, List

import matplotlib.pyplot as plt


from ._utils import _get_id
from ..plot import colors, scatter


class ComparerPlotter:
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
        from ._comparison import MOD_COLORS  # TODO move to here

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
                cmp.data[cmp._obs_name].values,
                marker=".",
                color=cmp.data[cmp._obs_name].attrs["color"],
            )
            ax.set_ylabel(cmp._unit_text)
            ax.legend([*cmp.mod_names, cmp._obs_name])
            ax.set_ylim(ylim)
            plt.title(title)
            return ax

        elif backend == "plotly":  # pragma: no cover
            import plotly.graph_objects as go

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
                        y=cmp.data[cmp._obs_name].values,
                        name=cmp._obs_name,
                        mode="markers",
                        marker=dict(color=cmp.data[cmp._obs_name].attrs["color"]),
                    ),
                ]
            )

            fig.update_layout(title=title, yaxis_title=cmp._unit_text, **kwargs)
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
        kwargs : other keyword arguments to df.hist()

        Returns
        -------
        matplotlib axes

        See also
        --------
        pandas.Series.hist
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

        cmp.data[cmp._obs_name].to_series().hist(
            bins=bins, color=cmp.data[cmp._obs_name].attrs["color"], ax=ax, **kwargs
        )
        ax.legend([mod_name, cmp._obs_name])
        plt.title(title)
        plt.xlabel(f"{cmp._unit_text}")
        if density:
            plt.ylabel("density")
        else:
            plt.ylabel("count")

        return ax

    def kde(self, ax=None, **kwargs):

        cmp = self.comparer

        if ax is None:
            ax = plt.gca()

        cmp.data.Observation.to_series().plot.kde(
            ax=ax, linestyle="dashed", label="Observation", **kwargs
        )

        for model in cmp.mod_names:
            cmp.data[model].to_series().plot.kde(ax=ax, label=model, **kwargs)

        ax.set_xlabel(cmp._unit_text)  # TODO

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
        quantiles: Union[int, List[float]] = None,
        fit_to_quantiles: bool = False,
        show_points: Union[bool, int, float] = None,
        show_hist: bool = None,
        show_density: bool = None,
        norm: colors = None,
        backend: str = "matplotlib",
        figsize: List[float] = (8, 8),
        xlim: List[float] = None,
        ylim: List[float] = None,
        reg_method: str = "ols",
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        skill_table: Union[str, List[str], bool] = None,
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
        >>> cmp.plot.scatter(model='HKZN_v2', figsize=(10, 10))
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

        unit_text = cmp._unit_text
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
