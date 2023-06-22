import matplotlib.pyplot as plt

from ._utils import _get_id


class ComparerPlotter:
    def __init__(self, comparer):
        self.comparer = comparer

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

    def scatter(self, **kwargs):
        pass


class ComparisonCollectionPlotter:
    pass
