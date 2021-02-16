from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from copy import deepcopy

from mikeio import Dfs0, Dataset
import mikefm_skill.metrics as mtr
from mikefm_skill.observation import PointObservation, TrackObservation


class BaseComparer:
    observation = None
    obs_name = "Observation"
    mod_names = None
    mod_colors = ["#004165", "#63CEFF", "#8B8D8E", "#0098DB", "#93509E", "#61C250"]
    resi_color = "#8B8D8E"
    #      darkblue: #004165
    #      midblue:  #0098DB,
    #      gray:     #8B8D8E,
    #      lightblue:#63CEFF,
    #      green:    #61C250
    #      purple:   #93509E
    mod_data = None
    df = None

    _mod_start = datetime(2900, 1, 1)
    _mod_end = datetime(1, 1, 1)

    @property
    def name(self):
        return self.observation.name

    @property
    def residual(self):
        # TODO
        return self.mod - np.vstack(self.obs)

    @property
    def obs(self):
        return self.df[self.obs_name].values

    @property
    def mod(self):
        return self.df[self.mod_names].values

    @property
    def n_models(self):
        return len(self.mod_names)

    def __init__(self, observation, modeldata=None, metric=mtr.rmse):
        self.observation = deepcopy(observation)
        self.mod_names = []
        self.mod_data = {}

        if modeldata is not None:
            self.add_modeldata(modeldata)

        self.metric = mtr.rmse

    def add_modeldata(self, modeldata):
        if isinstance(modeldata, list):
            for data in modeldata:
                self.add_modeldata(data)
            return

        if isinstance(modeldata, Dataset):
            mod_df = modeldata.to_dataframe()
        elif isinstance(modeldata, pd.DataFrame):
            # TODO: add validation
            mod_df = modeldata
        else:
            raise ValueError("Unknown modeldata type (mikeio.Dataset or pd.DataFrame)")
        mod_name = mod_df.columns[-1]
        self.mod_data[mod_name] = mod_df
        self.mod_names.append(mod_name)

        if mod_df.index[0] < self._mod_start:
            self._mod_start = mod_df.index[0].to_pydatetime()
        if mod_df.index[-1] > self._mod_end:
            self._mod_end = mod_df.index[-1].to_pydatetime()

    def __repr__(self):

        out = []
        out.append(f"<{type(self).__name__}>")
        out.append(f"Observation: {self.observation.name}")
        # out.append(f"{self.metric.__name__}: {self.skill():.3f}")
        return str.join("\n", out)

    def remove_bias(self, correct="Model"):
        bias = self.residual.mean(axis=0)
        if correct == "Model":
            for j in range(self.n_models):
                mod_name = self.mod_names[j]
                mod_df = self.mod_data[mod_name]
                mod_df[mod_name] = mod_df.values - bias[j]
            self.df[self.mod_names] = self.mod - bias
        elif correct == "Observation":
            # what if multiple models?
            self.df[self.obs_name] = self.obs + bias
        else:
            raise ValueError(
                f"Unknown correct={correct}. Only know 'Model' and 'Observation'"
            )
        return bias

    def _get_mod_id(self, model):
        mod_id = 0
        if self.n_models > 1 and model is str:
            if model in self.mod_names:
                mod_id = self.mod_names.index(model)
        if self.n_models > 1 and model is int:
            if model >= 0 and model < self.n_models:
                mod_id = model
        return mod_id

    def scatter(
        self,
        model=None,
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
        mod_id = self._get_mod_id(model)

        x = self.obs
        y = self.mod[:, mod_id]

        if xlabel is None:
            xlabel = f"Observation, {self.observation._unit_text()}"

        if ylabel is None:
            ylabel = f"Model, {self.observation._unit_text()}"

        if title is None:
            title = f"{self.mod_names[mod_id]} vs {self.name}"

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
                plt.scatter(x, y, c="0.25", s=20, alpha=0.5, marker=".", label=None)
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

    def hist(self, model=None, bins=100):
        mod_id = self._get_mod_id(model)
        mod_name = self.mod_names[mod_id]

        ax = self.df[mod_name].hist(bins=bins, color=self.mod_colors[mod_id], alpha=0.5)
        self.df[self.obs_name].hist(
            bins=bins, color=self.observation.color, alpha=0.5, ax=ax
        )
        ax.legend([mod_name, self.obs_name])
        plt.title(f"{mod_name} vs {self.name}")
        plt.xlabel(f"{self.observation._unit_text()}")

    def skill(self, model=None, metric=None):

        mod_id = self._get_mod_id(model)

        if metric is None:
            metric = self.metric

        return metric(self.obs, self.mod[:, mod_id])


class PointComparer(BaseComparer):
    def __init__(self, observation, modeldata):
        super().__init__(observation, modeldata)
        assert isinstance(observation, PointObservation)
        self.observation.df = self.observation.df[self._mod_start : self._mod_end]

        if not isinstance(modeldata, list):
            modeldata = [modeldata]
        for j, data in enumerate(modeldata):
            df = self._model2obs_interp(self.observation, data)
            if j == 0:
                self.df = df
            else:
                self.df[self.mod_names[j]] = df[self.mod_names[j]]

        self.df.dropna(inplace=True)

    def _model2obs_interp(self, obs, mod_ds):
        """interpolate model to measurement time"""
        df = mod_ds.interp_time(obs.time).to_dataframe()
        df[self.obs_name] = obs.values
        return df.iloc[:, ::-1]

    def plot_timeseries(self, title=None, figsize=None, backend="matplotlib", **kwargs):

        mod_df = self.mod_data[self.mod_names[0]]
        if title is None:
            title = self.name

        if backend == "matplotlib":
            _, ax = plt.subplots(figsize=figsize)
            for j in range(self.n_models):
                key = self.mod_names[j]
                self.mod_data[key].plot(ax=ax, color=self.mod_colors[j])

            ax.scatter(
                self.df.index,
                self.df[[self.obs_name]],
                marker=".",
                color=self.observation.color,
            )
            ax.set_ylabel(self.observation._unit_text())
            ax.legend([*self.mod_names, self.obs_name])

            plt.title(title)
            return ax

        elif backend == "plotly":
            import plotly.graph_objects as go

            mod_scatter_list = []
            for j in range(self.n_models):
                key = self.mod_names[j]
                mod_df = self.mod_data[key]
                mod_scatter_list.append(
                    go.Scatter(
                        x=mod_df.index,
                        y=mod_df.iloc[:, 0],
                        name=key,
                        line=dict(color=self.mod_colors[j]),
                    )
                )

            fig = go.Figure(
                [
                    *mod_scatter_list,
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
        self.observation.df = self.observation.df[self._mod_start : self._mod_end]

        if not isinstance(modeldata, list):
            modeldata = [modeldata]
        for j, data in enumerate(modeldata):
            df = data.to_dataframe()
            if j == 0:
                df[self.obs_name] = observation.df.iloc[:, -1].values
                self.df = df
            else:
                self.df[self.mod_names[j]] = df[self.mod_names[j]]

        self.df = df.dropna()
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

    def skill_report(self, model=None, metrics: list = None) -> pd.DataFrame:

        if metrics is None:
            metrics = [mtr.bias, mtr.rmse, mtr.corr_coef, mtr.scatter_index]

        res = {}
        for cmp in self.comparisons.values():
            mod_id = cmp._get_mod_id(model)
            tmp = {}
            for metric in metrics:
                tmp[metric.__name__] = metric(cmp.obs, cmp.mod[:, mod_id])
            res[cmp.name] = tmp

        return pd.DataFrame(res).T
