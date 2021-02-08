import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

from mikeio import Dfs0
import mikefm_skill.metrics as mtr
from mikefm_skill.observation import PointObservation, TrackObservation


class BaseComparer:
    observation = None
    mod_color = "#004165"
    resi_color = "#8B8D8E"
    mod_name = None
    obs_name = "Observation"
    mod_df = None
    df = None

    @property
    def name(self):
        return self.observation.name

    @property
    def residual(self):
        return self.mod - self.obs

    @property
    def obs(self):
        return self.df[self.obs_name].values

    @property
    def mod(self):
        return self.df[self.mod_name].values

    def __init__(self, observation, modeldata, metric=mtr.rmse):
        self.observation = deepcopy(observation)
        self.mod_df = modeldata.to_dataframe()
        self.mod_name = self.mod_df.columns[-1]
        self.metric = mtr.rmse

    def __repr__(self):

        out = []
        out.append(f"<{type(self).__name__}>")
        out.append(f"Observation: {self.observation.name}")
        out.append(f"{self.metric.__name__}: {self.skill():.3f}")
        return str.join("\n", out)

    def remove_bias(self, correct="Model"):
        bias = self.residual.mean()
        if correct == "Model":
            self.mod_df[self.mod_name] = self.mod_df.values - bias
            self.df[self.mod_name] = self.mod - bias
        elif correct == "Observation":
            self.df[self.obs_name] = self.obs + bias
        else:
            raise ValueError(
                f"Unknown correct={correct}. Only know 'Model' and 'Observation'"
            )
        return bias

    def scatter(
        self,
        xlabel=None,
        ylabel=None,
        binsize=None,
        nbins=100,
        show_points=None,
        backend="matplotlib",
        title=None,
        **kwargs,
    ):

        x = self.obs
        y = self.mod

        if xlabel is None:
            xlabel = f"Observation, {self.observation._unit_text()}"

        if ylabel is None:
            ylabel = f"Model, {self.observation._unit_text()}"

        if title is None:
            title = f"{self.mod_name} vs {self.name}"

        if show_points is None:
            show_points = len(x) < 1e4

        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        xymin = min([xmin, ymin])
        xymax = max([xmax, ymax])

        if binsize is None:
            binsize = (xmax - xmin) / nbins
        else:
            nbins = int((xmax - xmin) / binsize)

        xq = np.quantile(x, q=np.linspace(0, 1, num=nbins))
        yq = np.quantile(y, q=np.linspace(0, 1, num=nbins))

        if backend == "matplotlib":
            plt.plot([xymin, xymax], [xymin, xymax], label="1:1")
            plt.plot(xq, yq, label="QQ", c="gray")
            plt.hist2d(x, y, bins=nbins, cmin=0.01, **kwargs)
            plt.legend()
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.axis("square")
            plt.xlim([xymin, xymax])
            plt.ylim([xymin, xymax])
            cbar = plt.colorbar()
            cbar.set_label("# points")
            if show_points:
                plt.scatter(x, y, c="0.25", s=10, alpha=0.2, marker=".", label=None)
            plt.title(title)

        elif backend == "bokeh":
            import bokeh.plotting as bh
            import bokeh.models as bhm

            # bh.output_notebook()

            p = bh.figure(x_axis_label="obs", y_axis_label="model", title=title)
            p.hexbin(x, y, size=binsize)
            p.scatter(xq, yq, legend_label="Q-Q", color="gray")

            linvals = np.linspace(np.min([x, y]), np.max([x, y]))
            p.line(linvals, linvals, legend_label="1:1")
            # mder = bhm.Label(x=10, y=500, x_units='screen', y_units='screen',
            #                text=f"MdEr: {MdEr(x,y):.2f}")
            # rmse = bhm.Label(x=10, y=480, x_units='screen', y_units='screen',
            #                text=f"RMSE: {RMSE(x,y):.2f}")
            # stder = bhm.Label(x=10, y=460, x_units='screen', y_units='screen',
            #                text=f"StdEr: {StdEr(x,y):.2f}")
            # N = bhm.Label(x=10, y=440, x_units='screen', y_units='screen',
            #                text=f"N: {len(x)}")

            # p.add_layout(mder)
            # p.add_layout(rmse)
            # p.add_layout(stder)
            # p.add_layout(N)
            bh.show(p)

        else:
            raise ValueError(f"Plotting backend: {backend} not supported")

    def residual_hist(self, bins=100):
        plt.hist(self.residual, bins=bins, color=self.resi_color)
        plt.title(f"Residuals, {self.name}")
        plt.xlabel(f"Residuals of {self.observation._unit_text()}")

    def hist(self, bins=100):
        ax = self.df.iloc[:, -2].hist(bins=bins, color=self.mod_color, alpha=0.5)
        self.df.iloc[:, -1].hist(
            bins=bins, color=self.observation.color, alpha=0.5, ax=ax
        )
        ax.legend([self.mod_name, self.obs_name])
        plt.title(f"{self.mod_name} vs {self.name}")
        plt.xlabel(f"{self.observation._unit_text()}")

    def statistics(self):
        resi = self.residual
        bias = resi.mean()
        uresi = resi - bias
        rmse = np.sqrt(np.mean(uresi ** 2))
        return bias, rmse

    def skill(self, metric=None):

        if metric is None:
            metric = self.metric

        return metric(self.obs, self.mod)


# TODO: add more ModelResults
class PointComparer(BaseComparer):
    def __init__(self, observation, modeldata):
        super().__init__(observation, modeldata)
        assert isinstance(observation, PointObservation)
        self.observation.df = self.observation.df[
            modeldata.start_time : modeldata.end_time
        ]
        self.df = self._model2obs_interp(self.observation, modeldata)

    def _model2obs_interp(self, obs, mod_ds):
        """interpolate model to measurement time"""
        df = mod_ds.interp_time(obs.time).to_dataframe()
        df[self.obs_name] = obs.values
        return df

    def plot_timeseries(self, title=None, figsize=None):
        ax = self.mod_df.plot(figsize=figsize)
        ax.scatter(
            self.df.index,
            self.df[[self.obs_name]],
            marker=".",
            color=self.observation.color,
        )
        ax.set_ylabel(self.observation._unit_text())
        ax.legend([self.mod_name, self.obs_name])
        if title is None:
            title = self.name
        plt.title(title)


class TrackComparer(BaseComparer):
    def __init__(self, observation, modeldata):
        super().__init__(observation, modeldata)
        assert isinstance(observation, TrackObservation)
        self.observation.df = self.observation.df[
            modeldata.start_time : modeldata.end_time
        ]
        self.df = modeldata.to_dataframe()
        self.df[self.obs_name] = observation.df.iloc[:, -1].values
        self.df.dropna(inplace=True)
        # TODO: add check


class ComparisonCollection:
    def __init__(self):
        self.comparisons = []

    def __getitem__(self, x):
        return self.comparisons[x]

    def add_comparison(self, comparison):

        self.comparisons.append(comparison)

    def skill(self, metric=mtr.rmse):

        scores = [metric(mr.obs, mr.mod) for mr in self.comparisons]

        return np.average(scores)

    def skill_report(self, metrics: list = None) -> pd.DataFrame:

        if metrics is None:
            metrics = [mtr.bias, mtr.rmse, mtr.corr_coef, mtr.scatter_index]

        res = {}
        for mr in self.comparisons:
            tmp = {}
            for metric in metrics:
                tmp[metric.__name__] = metric(mr.obs, mr.mod)
            res[mr.name] = tmp

        return pd.DataFrame(res).T
