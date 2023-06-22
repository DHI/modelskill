import matplotlib.pyplot as plt

from ._utils import _get_id


class ComparerPlotter:
    def __init__(self, comparer):
        self.comparer = comparer

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

    def timeseries(self, **kwargs):
        pass


class ComparisonCollectionPlotter:
    pass
