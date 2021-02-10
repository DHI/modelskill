from matplotlib import markers
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
        nbins=20,
        show_points=None,
        backend="matplotlib",
        title=None,
        figsize=(8, 8),
        xlim=None,
        ylim=None,
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

        if xlim is None:
            xlim = [xymin, xymax]

        if ylim is None:
            ylim = [xymin, xymax]

        if binsize is None:
            binsize = (xmax - xmin) / nbins
        else:
            nbins = int((xmax - xmin) / binsize)

        xq = np.quantile(x, q=np.linspace(0, 1, num=nbins))
        yq = np.quantile(y, q=np.linspace(0, 1, num=nbins))

        if backend == "matplotlib":

            plt.figure(figsize=figsize)
            plt.plot([xymin, xymax], [xymin, xymax], label="1:1", c="blue")
            plt.plot(xq, yq, label="QQ", c="gray")
            plt.hist2d(x, y, bins=nbins, cmin=0.01, **kwargs)
            plt.legend()
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.axis("square")
            plt.xlim(xlim)
            plt.ylim(ylim)
            cbar = plt.colorbar(shrink=0.6, pad=0.01)
            cbar.set_label("# points")
            if show_points:
                plt.scatter(x, y, c="0.25", s=10, alpha=0.2, marker=".", label=None)
            plt.title(title)

        elif backend == "bokeh":
            import bokeh.plotting as bh
            import bokeh.models as bhm

            p = bh.figure(x_axis_label=xlabel, y_axis_label=ylabel, title=title)
            p.hexbin(x, y, size=binsize)
            p.line(xq, yq, legend_label="Q-Q", color="gray", line_width=2)

            linvals = np.linspace(np.min([x, y]), np.max([x, y]))
            p.line(linvals, linvals, legend_label="1:1", line_width=2, color="blue")

            bh.show(p)
        elif backend == "plotly":
            import plotly.graph_objects as go

            linvals = np.linspace(np.min([x, y]), np.max([x, y]))

            data = [
                go.Scatter(x=linvals, y=linvals, name="1:1", line=dict(color="blue")),
                go.Scatter(x=xq, y=yq, name="Q-Q", line=dict(color="gray")),
                go.Histogram2d(
                    x=x,
                    y=y,
                    xbins=dict(size=binsize),
                    ybins=dict(size=binsize),
                    colorscale=[
                        [0.0, "rgba(0,0,0,0)"],
                        [0.1, "purple"],
                        [0.5, "green"],
                        [1.0, "yellow"],
                    ],
                ),
            ]

            if show_points:
                data.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        name="Data",
                        marker=dict(color="black"),
                    )
                )

            defaults = {"width": 600, "height": 600}
            defaults = {**defaults, **kwargs}

            layout = layout = go.Layout(
                legend=dict(x=0.01, y=0.99),
                yaxis=dict(scaleanchor="x", scaleratio=1),
                title=dict(text=title, xanchor="center", yanchor="top", x=0.5, y=0.9),
                yaxis_title=ylabel,
                xaxis_title=xlabel,
                **defaults,
            )

            fig = go.Figure(data=data, layout=layout)
            fig.update_xaxes(range=xlim)
            fig.update_yaxes(range=ylim)
            fig.show()

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

    def plot_timeseries(self, title=None, figsize=None, backend="matplotlib", **kwargs):

        if backend == "matplotlib":
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

        elif backend == "plotly":
            import plotly.graph_objects as go

            fig = go.Figure(
                [
                    go.Scatter(
                        x=self.mod_df.index,
                        y=self.mod_df.iloc[:, 0],
                        name=self.mod_name,
                        line=dict(color=self.mod_color),
                    ),
                    go.Scatter(
                        x=self.df.index,
                        y=self.df[self.obs_name],
                        name=self.obs_name,
                        mode="markers",
                        marker=dict(color=self.observation.color),
                    ),
                ]
            )

            fig.update_layout(
                title=title, yaxis_title=self.observation._unit_text(), **kwargs
            )

            fig.show()
        else:
            raise ValueError(f"Plotting backend: {backend} not supported")


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
        self.comparisons = {}

    def __repr__(self):
        out = []
        out.append(f"<{type(self).__name__}>")
        for key, value in self.comparisons.items():
            out.append(f"{type(value).__name__}: {key}")
        return str.join("\n", out)

    def __getitem__(self, x):
        return self.comparisons[x]

    def add_comparison(self, comparison):

        self.comparisons[comparison.name] = comparison

    def skill(self, metric=mtr.rmse):

        scores = [metric(mr.obs, mr.mod) for mr in self.comparisons.values()]

        return np.average(scores)

    def skill_report(self, metrics: list = None) -> pd.DataFrame:

        if metrics is None:
            metrics = [mtr.bias, mtr.rmse, mtr.corr_coef, mtr.scatter_index]

        res = {}
        for cmp in self.comparisons.values():
            tmp = {}
            for metric in metrics:
                tmp[metric.__name__] = metric(cmp.obs, cmp.mod)
            res[cmp.name] = tmp

        return pd.DataFrame(res).T
