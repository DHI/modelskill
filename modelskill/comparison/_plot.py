import matplotlib.pyplot as plt


class ComparerPlotter:
    def __init__(self, comparer):
        self.comparer = comparer

    def hist(self, **kwargs):
        pass

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
