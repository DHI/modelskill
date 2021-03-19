"""The `compare` module contains different types of comparer classes for
fixed locations (PointComparer), or locations moving in space (TrackComparer).

These Comparers are constructed by extracting data from the combination of observation and model results

Examples
--------
>>> mr = ModelResult("Oresund2D.dfsu")
>>> o1 = PointObservation("klagshamn.dfs0, item=0, x=366844, y=6154291, name="Klagshamn")
>>> mr.add_observation(o1, item=0)
>>> comparer = mr.extract()
"""
from collections.abc import Mapping
from typing import List
from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from copy import deepcopy
from scipy.stats import linregress
from scipy import odr

from mikeio import Dfs0, Dataset
import fmskill.metrics as mtr
from fmskill.observation import PointObservation, TrackObservation


class BaseComparer:
    """Abstract base class for all comparers, only used to inherit from, not to be used directly"""

    # observation = None
    obs_name = "Observation"
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
    def n(self) -> int:
        return len(self.df)

    @property
    def start(self) -> datetime:
        return self.df.index[0].to_pydatetime()

    @property
    def end(self) -> datetime:
        return self.df.index[-1].to_pydatetime()

    @property
    def x(self) -> float:
        return self.observation.x

    @property
    def y(self) -> float:
        return self.observation.y

    @property
    def name(self) -> str:
        return self.observation.name

    @property
    def residual(self):
        # TODO
        return self.mod - np.vstack(self.obs)

    @property
    def obs(self) -> np.ndarray:
        return self.df[self.obs_name].values

    @property
    def mod(self) -> np.ndarray:
        return self.df[self.mod_names].values

    @property
    def n_models(self) -> int:
        return len(self.mod_names)

    @property
    def mod_names(self) -> List[str]:
        return list(self.mod_data.keys())

    def __init__(self, observation, modeldata=None):
        self.observation = deepcopy(observation)
        self.mod_data = {}

        if modeldata is not None:
            self.add_modeldata(modeldata)

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
        if model is None or self.n_models <= 1:
            return 0
        elif isinstance(model, str):
            if model in self.mod_names:
                mod_id = self.mod_names.index(model)
            else:
                raise ValueError(
                    f"model {model} could not be found in {self.mod_names}"
                )
        elif isinstance(model, int):
            if model >= 0 and model < self.n_models:
                mod_id = model
            else:
                raise ValueError(
                    f"model id was {model} - must be within 0 and {self.n_models-1}"
                )
        else:
            raise ValueError("model must be None, str or int")
        return mod_id

    def scatter(
        self,
        model=None,
        xlabel=None,
        ylabel=None,
        binsize=None,
        nbins=20,
        show_points=None,
        show_hist=True,
        backend="matplotlib",
        title=None,
        figsize=(8, 8),
        xlim=None,
        ylim=None,
        reg_method="ols",
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

        # linear fit
        if reg_method == "ols":
            reg = linregress(x, y)
            intercept = reg.intercept
            slope = reg.slope
        elif reg_method == "odr":
            data = odr.Data(x, y)
            odr_obj = odr.ODR(data, odr.unilinear)
            output = odr_obj.run()

            intercept = output.beta[1]
            slope = output.beta[0]
        else:
            raise NotImplementedError(
                f"Regression method: {reg_method} not implemented, select 'ols' or 'odr'"
            )

        if intercept < 0:
            sign = ""
        else:
            sign = "+"
        reglabel = f"Fit: y={slope:.2f}x{sign}{intercept:.2f}"

        if backend == "matplotlib":

            plt.figure(figsize=figsize)
            plt.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], label="1:1", c="blue")
            plt.plot(xq, yq, label="QQ", c="gray")
            plt.plot(
                x,
                intercept + slope * x,
                "r",
                label=reglabel,
            )
            if show_hist:
                plt.hist2d(x, y, bins=nbins, cmin=0.01, **kwargs)
            plt.legend()
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.axis("square")
            plt.xlim(xlim)
            plt.ylim(ylim)
            if show_hist:
                cbar = plt.colorbar(shrink=0.6, pad=0.01)
                cbar.set_label("# points")
            if show_points:
                plt.scatter(x, y, c="0.25", s=20, alpha=0.5, marker=".", label=None)
            plt.title(title)

        elif backend == "plotly":
            import plotly.graph_objects as go

            linvals = np.linspace(np.min([x, y]), np.max([x, y]))

            data = [
                go.Scatter(
                    x=x,
                    y=intercept + slope * x,
                    name=reglabel,
                    mode="lines",
                    line=dict(color="red"),
                ),
                go.Scatter(
                    x=xlim, y=xlim, name="1:1", mode="lines", line=dict(color="blue")
                ),
                go.Scatter(
                    x=xq, y=yq, name="Q-Q", mode="lines", line=dict(color="gray")
                ),
            ]

            if show_hist:
                data.append(
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
                    )
                )

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
            fig.show()  # Should this be here

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
            metric = mtr.rmse

        return metric(self.obs, self.mod[:, mod_id])


class PointComparer(BaseComparer):
    """
    Comparer for observations from fixed locations

    Examples
    --------
    >>> mr = ModelResult("Oresund2D.dfsu")
    >>> o1 = PointObservation("klagshamn.dfs0, item=0, x=366844, y=6154291, name="Klagshamn")
    >>> mr.add_observation(o1, item=0)
    >>> comparer = mr.extract()
    """

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

    def plot_timeseries(
        self, title=None, ylim=None, figsize=None, backend="matplotlib", **kwargs
    ):

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
            ax.set_ylim(ylim)
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
            fig.update_yaxes(range=ylim)

            fig.show()
        else:
            raise ValueError(f"Plotting backend: {backend} not supported")


class TrackComparer(BaseComparer):
    @property
    def x(self):
        return self.df.iloc[:, 0]

    @property
    def y(self):
        return self.df.iloc[:, 1]

    def __init__(self, observation, modeldata):
        super().__init__(observation, modeldata)
        assert isinstance(observation, TrackObservation)
        self.observation.df = self.observation.df[self._mod_start : self._mod_end]

        if not isinstance(modeldata, list):
            modeldata = [modeldata]
        for j, data in enumerate(modeldata):
            df = data.to_dataframe()
            if j == 0:
                df[self.obs_name] = observation.df.iloc[:, -1]
                cols = list(df.columns)
                cols = list((*cols[0:2], *cols[:1:-1]))
                self.df = df[cols]
            else:

                self.df[self.mod_names[j]] = df[self.mod_names[j]]

        self.df = self.df.dropna()
        # TODO: add check


class ComparisonCollection(Mapping):
    """
    Collection of comparers, constructed by calling the `ModelResult.extract` method.

    Examples
    --------
    >>> mr = ModelResult("Oresund2D.dfsu")
    >>> o1 = PointObservation("klagshamn.dfs0, item=0, x=366844, y=6154291, name="Klagshamn")
    >>> o2 = PointObservation("drogden.dfs0", item=0, x=355568.0, y=6156863.0)
    >>> mr.add_observation(o1, item=0)
    >>> mr.add_observation(o2, item=0)
    >>> comparer = mr.extract()

    """

    _all_df = None
    _mod_names: List[str]
    _obs_names: List[str]
    _start = datetime(2900, 1, 1)
    _end = datetime(1, 1, 1)
    _n_points = 0

    @property
    def n_points(self) -> int:
        return self._n_points

    @property
    def start(self) -> datetime:
        return self._start

    @property
    def end(self) -> datetime:
        return self._end

    @property
    def n_comparisons(self) -> int:
        return len(self.comparisons)

    @property
    def n_models(self) -> int:
        return len(set(self._mod_names))  # TODO why are there duplicates here

    @property
    def all_df(self) -> pd.DataFrame:
        if self._all_df is None:
            self._construct_all_df()
        return self._all_df

    def _construct_all_df(self):
        # TODO: var_name
        if self.n_comparisons == 0:
            return

        template = {
            "mod_name": pd.Series([], dtype="str"),
            "obs_name": pd.Series([], dtype="str"),
            "x": pd.Series([], dtype="float"),
            "y": pd.Series([], dtype="float"),
            "mod_val": pd.Series([], dtype="float"),
            "obs_val": pd.Series([], dtype="float"),
        }
        cols = template.keys()
        res = pd.DataFrame(template)

        for cmp in self.comparisons.values():
            for j in range(cmp.n_models):
                mod_name = cmp.mod_names[j]
                df = cmp.df[[mod_name]].copy()
                df.columns = ["mod_val"]
                df["mod_name"] = mod_name
                df["obs_name"] = cmp.observation.name
                df["x"] = cmp.x
                df["y"] = cmp.y
                df["obs_val"] = cmp.obs
                res = res.append(df[cols])

        self._all_df = res.sort_index()

    def __init__(self):
        self.comparisons = {}
        self._mod_names = []
        self._obs_names = []

    def __repr__(self):
        out = []
        out.append(f"<{type(self).__name__}>")
        for key, value in self.comparisons.items():
            out.append(f"{type(value).__name__}: {key}")
        return str.join("\n", out)

    def __getitem__(self, x):
        return self.comparisons[x]

    def __len__(self) -> int:
        return len(self.comparisons)

    def __iter__(self):
        return iter(self.comparisons)

    def add_comparison(self, comparison: BaseComparer):

        self.comparisons[comparison.name] = comparison
        for mod_name in comparison.mod_names:
            if mod_name not in self._mod_names:
                self._mod_names.append(mod_name)
        self._n_points = self._n_points + comparison.n
        if comparison.start < self.start:
            self._start = comparison.start
        if comparison.end > self.end:
            self._end = comparison.end

        self._all_df = None

    def compound_skill(self, metric=mtr.rmse) -> float:
        """Compound skill (possibly weighted)"""
        cmps = self.comparisons.values()
        scores = [metric(mr.obs, mr.mod) for mr in cmps]
        weights = [c.observation.weight for c in cmps]
        return np.average(scores, weights=weights)

    def skill_df(
        self,
        df=None,
        model=None,
        observation=None,
        start=None,
        end=None,
        metrics: list = None,
    ) -> pd.DataFrame:

        if metrics is None:
            metrics = [mtr.bias, mtr.rmse, mtr.mape, mtr.cc, mtr.si, mtr.r2]

        if df is None:
            df = self.sel_df(model=model, observation=observation, start=start, end=end)
        mod_names = df.mod_name.unique()
        obs_names = df.obs_name.unique()

        rows = []
        for mod_name in mod_names:
            for obs_name in obs_names:
                dfsub = df[(df.mod_name == mod_name) & (df.obs_name == obs_name)]
                row = {}
                row["model"] = mod_name
                row["observation"] = obs_name
                row["n"] = len(dfsub)
                for metric in metrics:
                    row[metric.__name__] = metric(
                        dfsub.obs_val.values, dfsub.mod_val.values
                    )
                rows.append(row)
        res = pd.DataFrame(rows)

        if len(mod_names) == 1:
            res.index = res.observation
            res.drop(columns=["observation", "model"], inplace=True)
        elif len(obs_names) == 1:
            res.index = res.model
            res.drop(columns=["observation", "model"], inplace=True)

        return res

    def sel_df(self, model=None, observation=None, start=None, end=None):
        # TODO: area
        df = self.all_df
        if model is not None:
            model = [model] if isinstance(model, str) else model
            df = df[df.mod_name.isin(model)]
        if observation is not None:
            observation = [observation] if isinstance(observation, str) else observation
            df = df[df.obs_name.isin(observation)]
        if (start is not None) or (end is not None):
            df = df.loc[start:end]

        return df

    def skill_report(self, model=None, metrics: list = None) -> pd.DataFrame:
        """Skill for each observation, weights are not taken into account"""

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
