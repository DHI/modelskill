"""The `compare` module contains different types of comparer classes for
fixed locations (PointComparer), or locations moving in space (TrackComparer).

These Comparers are constructed by extracting data from the combination of observation and model results

Examples
--------
>>> mr = ModelResult("Oresund2D.dfsu")
>>> o1 = PointObservation("klagshamn.dfs0", item=0, x=366844, y=6154291, name="Klagshamn")
>>> mr.add_observation(o1, item=0)
>>> comparer = mr.extract()
"""
from collections.abc import Mapping
from typing import List, Union
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
    _obs_names: List[str]
    _mod_names: List[str]
    _mod_colors = [
        "#004165",
        "#63CEFF",
        "#8B8D8E",
        "#0098DB",
        "#93509E",
        "#61C250",
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
        "#003f5c",
        "#2f4b7c",
        "#665191",
    ]

    _resi_color = "#8B8D8E"
    _obs_unit_text = ""
    #      darkblue: #004165
    #      midblue:  #0098DB,
    #      gray:     #8B8D8E,
    #      lightblue:#63CEFF,
    #      green:    #61C250
    #      purple:   #93509E
    mod_data = None
    df = None
    _all_df = None

    _mod_start = datetime(2900, 1, 1)
    _mod_end = datetime(1, 1, 1)

    @property
    def n_points(self) -> int:
        """number of compared points"""
        return len(self.df)

    @property
    def start(self) -> datetime:
        """start datetime of compared data"""
        return self.df.index[0].to_pydatetime()

    @property
    def end(self) -> datetime:
        """end datetime of compared data"""
        return self.df.index[-1].to_pydatetime()

    @property
    def x(self) -> float:
        return self.observation.x

    @property
    def y(self) -> float:
        return self.observation.y

    @property
    def name(self) -> str:
        """name of comparer (=observation)"""
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
        return self._mod_names  # list(self.mod_data.keys())

    @property
    def all_df(self) -> pd.DataFrame:
        if self._all_df is None:
            self._construct_all_df()
        return self._all_df

    def _all_df_template(self):
        template = {
            "mod_name": pd.Series([], dtype="category"),
            "obs_name": pd.Series([], dtype="category"),
            "x": pd.Series([], dtype="float"),
            "y": pd.Series([], dtype="float"),
            "mod_val": pd.Series([], dtype="float"),
            "obs_val": pd.Series([], dtype="float"),
        }
        res = pd.DataFrame(template)
        return res

    def _construct_all_df(self):
        # TODO: var_name
        res = self._all_df_template()
        cols = res.keys()
        for j in range(self.n_models):
            mod_name = self.mod_names[j]
            df = self.df[[mod_name]].copy()
            df.columns = ["mod_val"]
            df["mod_name"] = mod_name
            df["obs_name"] = self.observation.name
            df["x"] = self.x
            df["y"] = self.y
            df["obs_val"] = self.obs
            res = res.append(df[cols])

        self._all_df = res.sort_index()

    def __init__(self, observation, modeldata=None):
        self.observation = deepcopy(observation)
        self._obs_unit_text = self.observation._unit_text()
        self.mod_data = {}
        self._obs_names = [observation.name]

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
        self._mod_names = list(self.mod_data.keys())

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

    def _get_obs_name(self, obs):
        return self._obs_names[self._get_obs_id(obs)]

    def _get_obs_id(self, obs):
        if obs is None or self.n_observations <= 1:
            return 0
        elif isinstance(obs, str):
            if obs in self._obs_names:
                obs_id = self._obs_names.index(obs)
            else:
                raise ValueError(f"obs {obs} could not be found in {self._obs_names}")
        elif isinstance(obs, int):
            if obs >= 0 and obs < self.n_observations:
                obs_id = obs
            else:
                raise ValueError(
                    f"obs id was {obs} - must be within 0 and {self.n_observations-1}"
                )
        else:
            raise ValueError("observation must be None, str or int")
        return obs_id

    def _get_mod_name(self, model):
        return self._mod_names[self._get_mod_id(model)]

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

    def skill(
        self,
        model: Union[str, int, List[str], List[int]] = None,
        observation: Union[str, int, List[str], List[int]] = None,
        start: Union[str, datetime] = None,
        end: Union[str, datetime] = None,
        area: List[float] = None,
        df: pd.DataFrame = None,
        metrics: list = None,
    ) -> pd.DataFrame:
        """Skill assessment of model(s)

        Parameters
        ----------
        metrics : list, optional
            list of fmskill.metrics, by default [bias, rmse, urmse, mae, cc, si, r2]
        model : (str, int, List[str], List[int]), optional 
            name or ids of models to be compared, by default all
        observation : (str, int, List[str], List[int])), optional
            name or ids of observations to be compared, by default all
        start : (str, datetime), optional
            start time of comparison, by default None
        end : (str, datetime), optional
            end time of comparison, by default None
        area : list(float), optional
            bbox coordinates [x0, y0, x1, y1], 
            or polygon coordinates [x0, y0, x1, y1, ..., xn, yn], 
            by default None
        df : pd.dataframe, optional
            show user-provided data instead of the comparers own data, by default None
        
        Returns
        -------
        pd.DataFrame
            skill assessment as a dataframe

        See also
        --------
        sel_df
            a method for filtering/selecting data

        Examples
        --------
        >>> cc = mr.extract()
        >>> cc.skill().round(2)
                       n  bias  rmse  urmse   mae    cc    si    r2
        observation                                                
        HKNA         385 -0.20  0.35   0.29  0.25  0.97  0.09  0.99
        EPL           66 -0.08  0.22   0.20  0.18  0.97  0.07  0.99
        c2           113 -0.00  0.35   0.35  0.29  0.97  0.12  0.99

        >>> cc.skill(observation='c2', start='2017-10-28').round(2)
                       n  bias  rmse  urmse   mae    cc    si    r2
        observation                                               
        c2            41  0.33  0.41   0.25  0.36  0.96  0.06  0.99
        """

        if metrics is None:
            metrics = [mtr.bias, mtr.rmse, mtr.urmse, mtr.mae, mtr.cc, mtr.si, mtr.r2]

        df = self.sel_df(
            model=model, observation=observation, start=start, end=end, area=area, df=df
        )

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

    def sel_df(
        self,
        model: Union[str, int, List[str], List[int]] = None,
        observation: Union[str, int, List[str], List[int]] = None,
        start: Union[str, datetime] = None,
        end: Union[str, datetime] = None,
        area: List[float] = None,
        df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Select/filter data from all the compared data. 
        Used by compare.scatter and compare.skill to select data.

        Parameters
        ----------
        model : (str, int, List[str], List[int]), optional 
            name or ids of models to be compared, by default all
        observation : (str, int, List[str], List[int])), optional
            name or ids of observations to be compared, by default all
        start : (str, datetime), optional
            start time of comparison, by default None
        end : (str, datetime), optional
            end time of comparison, by default None
        area : list(float), optional
            bbox coordinates [x0, y0, x1, y1], 
            or polygon coordinates [x0, y0, x1, y1, ..., xn, yn], 
            by default None
        df : pd.dataframe, optional
            show user-provided data instead of the comparers own data, by default None

        Returns
        -------
        pd.DataFrame
            selected data in a dataframe with columns (mod_name,obs_name,x,y,mod_val,obs_val)

        See also
        --------
        skill 
            a method for aggregated skill assessment
        scatter
            a method for plotting compared data

        Examples
        --------
        >>> cc = mr.extract()        
        >>> dfsub = cc.sel_df(observation=['EPL','HKNA'])
        >>> dfsub = cc.sel_df(model=0)        
        >>> dfsub = cc.sel_df(start='2017-10-1', end='2017-11-1')
        >>> dfsub = cc.sel_df(area=[0.5,52.5,5,54])

        >>> cc.sel_df(observation='c2', start='2017-10-28').head(3)
                         mod_name obs_name      x       y   mod_val  obs_val
        2017-10-28 01:00:00  SW_1      EPL  3.276  51.999  1.644092     1.82
        2017-10-28 02:00:00  SW_1      EPL  3.276  51.999  1.755809     1.86
        2017-10-28 03:00:00  SW_1      EPL  3.276  51.999  1.867526     2.11
        """
        if df is None:
            df = self.all_df
        if model is not None:
            models = [model] if np.isscalar(model) else model
            models = [self._get_mod_name(m) for m in models]
            df = df[df.mod_name.isin(models)]
        if observation is not None:
            observation = [observation] if np.isscalar(observation) else observation
            observation = [self._get_obs_name(o) for o in observation]
            df = df[df.obs_name.isin(observation)]
        if (start is not None) or (end is not None):
            df = df.loc[start:end]
        if area is not None:
            if self._area_is_bbox(area):
                x0, y0, x1, y1 = area
                df = df[(df.x > x0) & (df.x < x1) & (df.y > y0) & (df.y < y1)]
            elif self._area_is_polygon(area):
                polygon = np.array(area)
                xy = np.column_stack((df.x.values, df.y.values))
                mask = self._inside_polygon(polygon, xy)
                df = df[mask]
            else:
                raise ValueError("area supports bbox [x0,y0,x1,y1] and closed polygon")
        return df

    def _area_is_bbox(self, area):
        is_bbox = False
        if area is not None:
            if not np.isscalar(area):
                area = np.array(area)
                if (area.ndim == 1) & (len(area) == 4):
                    if np.all(np.isreal(area)):
                        is_bbox = True
        return is_bbox

    def _area_is_polygon(self, area):
        if area is None:
            return False
        if np.isscalar(area):
            return False
        if not np.all(np.isreal(area)):
            return False
        polygon = np.array(area)
        if polygon.ndim > 2:
            return False

        if polygon.ndim == 1:
            if len(polygon) <= 7:
                return False
            x0, y0 = polygon[0:2]
            x1, y1 = polygon[-2:]
        if polygon.ndim == 2:
            if polygon.shape[0] <= 3:
                return False
            if polygon.shape[1] != 2:
                return False
            x0, y0 = polygon[0, :]
            x1, y1 = polygon[-1, :]

        if (x0 != x1) | (y0 != y1):
            # start point must equal end point
            return False

        return True

    def _inside_polygon(self, polygon, xy):
        import matplotlib.path as mp

        if polygon.ndim == 1:
            polygon = np.column_stack((polygon[0::2], polygon[1::2]))
        return mp.Path(polygon).contains_points(xy)

    def scatter(
        self,
        binsize: float = None,
        nbins: int = 20,
        show_points: bool = None,
        show_hist: bool = True,
        backend: str = "matplotlib",
        figsize: List[float] = (8, 8),
        xlim: List[float] = None,
        ylim: List[float] = None,
        reg_method: str = "ols",
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        model: Union[str, int] = None,
        observation: Union[str, int, List[str], List[int]] = None,
        start: Union[str, datetime] = None,
        end: Union[str, datetime] = None,
        area: List[float] = None,
        df: pd.DataFrame = None,
        **kwargs,
    ):
        """Scatter plot showing compared data: observation vs modelled
        Optionally, with density histogram. 

        Parameters
        ----------
        binsize : float, optional
            the size of each bin in the 2d histogram, by default None
        nbins : int, optional
            number of bins (if binsize is not given), by default 20
        show_points : bool, optional
            Should the scatter points be displayed?
            None means: only show points if fewer than threshold, by default None
        show_hist : bool, optional
            show the data density as a a 2d histogram, by default True
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
        model : (int, str), optional
            name or id of model to be compared, by default None
        observation : (int, str), optional
            name or ids of observations to be compared, by default None
        start : (str, datetime), optional
            start time of comparison, by default None
        end : (str, datetime), optional
            end time of comparison, by default None
        area : list(float), optional
            bbox coordinates [x0, y0, x1, y1], 
            or polygon coordinates[x0, y0, x1, y1, ..., xn, yn], 
            by default None
        df : pd.dataframe, optional
            show user-provided data instead of the comparers own data, by default None
        kwargs
        
        Examples
        ------
        >>> comparer.scatter()
        >>> comparer.scatter(binsize=0.2, backend='plotly')
        >>> comparer.scatter(show_points=False, title='no points')
        >>> comparer.scatter(xlabel='all observations', ylabel='my model')
        >>> comparer.scatter(model='HKZN_v2', figsize=(10, 10))
        >>> comparer.scatter(observations=['c2','HKNA'])
        """
        mod_id = self._get_mod_id(model)
        mod_name = self._mod_names[mod_id]

        df = self.sel_df(
            df=df,
            model=mod_name,
            observation=observation,
            start=start,
            end=end,
            area=area,
        )
        if len(df) == 0:
            raise Exception("No data found in selection")

        x = df.obs_val
        y = df.mod_val

        if xlabel is None:
            xlabel = f"Observation, {self._obs_unit_text}"

        if ylabel is None:
            ylabel = f"Model, {self._obs_unit_text}"

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
            plt.plot(xq, yq, label="Q-Q", c="gray")
            plt.plot(
                x, intercept + slope * x, "r", label=reglabel,
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
                cbar = plt.colorbar(fraction=0.046, pad=0.04)
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


class SingleObsComparer(BaseComparer):
    def skill(
        self,
        model: Union[str, int, List[str], List[int]] = None,
        start: Union[str, datetime] = None,
        end: Union[str, datetime] = None,
        area: List[float] = None,
        df: pd.DataFrame = None,
        metrics: list = None,
    ) -> pd.DataFrame:
        """Skill assessment of model(s)

        Parameters
        ----------
        metrics : list, optional
            list of fmskill.metrics, by default [bias, rmse, urmse, mae, cc, si, r2]
        model : (str, int, List[str], List[int]), optional 
            name or ids of models to be compared, by default all
        start : (str, datetime), optional
            start time of comparison, by default None
        end : (str, datetime), optional
            end time of comparison, by default None
        area : list(float), optional
            bbox coordinates [x0, y0, x1, y1], 
            or polygon coordinates [x0, y0, x1, y1, ..., xn, yn], 
            by default None
        df : pd.dataframe, optional
            show user-provided data instead of the comparers own data, by default None
        
        Returns
        -------
        pd.DataFrame
            skill assessment as a dataframe

        See also
        --------
        sel_df
            a method for filtering/selecting data

        Examples
        --------
        >>> cc = mr.extract()
        >>> cc['c2'].skill().round(2)
                       n  bias  rmse  urmse   mae    cc    si    r2
        observation                                                
        c2           113 -0.00  0.35   0.35  0.29  0.97  0.12  0.99
        """
        # only for improved documentation
        return super().skill(
            model=model, start=start, end=end, area=area, df=df, metrics=metrics
        )

    def score(
        self,
        metric=mtr.rmse,
        model: Union[str, int, List[str], List[int]] = None,
        start: Union[str, datetime] = None,
        end: Union[str, datetime] = None,
        area: List[float] = None,
        df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Model skill score

        Parameters
        ----------
        metric : list, optional
            a single metric from fmskill.metrics, by default rmse
        model : (str, int, List[str], List[int]), optional 
            name or ids of models to be compared, by default all
        start : (str, datetime), optional
            start time of comparison, by default None
        end : (str, datetime), optional
            end time of comparison, by default None
        area : list(float), optional
            bbox coordinates [x0, y0, x1, y1], 
            or polygon coordinates [x0, y0, x1, y1, ..., xn, yn], 
            by default None
        df : pd.dataframe, optional
            show user-provided data instead of the comparers own data, by default None

        Returns
        -------
        float
            skill score as a single number (for each model)

        See also
        --------
        skill
            a method for skill assessment returning a pd.DataFrame

        Examples
        --------
        >>> cc = mr.extract()
        >>> cc['c2'].score()
        0.3517964910888918
        
        >>> import fmskill.metrics as mtr
        >>> cc['c2'].score(metric=mtr.mape)
        11.567399646108198
        """

        df = self.skill(
            metrics=[metric], model=model, start=start, end=end, area=area, df=df,
        )
        values = df[metric.__name__].values
        if len(values) == 1:
            values = values[0]
        return values

    def sel_df(
        self,
        model: Union[str, int, List[str], List[int]] = None,
        observation: Union[str, int, List[str], List[int]] = None,
        start: Union[str, datetime] = None,
        end: Union[str, datetime] = None,
        area: List[float] = None,
        df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Select/filter data from all the compared data. 
        Used by compare.scatter and compare.skill to select data.

        Parameters
        ----------
        model : (str, int, List[str], List[int]), optional 
            name or ids of models to be compared, by default all
        start : (str, datetime), optional
            start time of comparison, by default None
        end : (str, datetime), optional
            end time of comparison, by default None
        area : list(float), optional
            bbox coordinates [x0, y0, x1, y1], 
            or polygon coordinates [x0, y0, x1, y1, ..., xn, yn], 
            by default None
        df : pd.dataframe, optional
            show user-provided data instead of the comparers own data, by default None

        Returns
        -------
        pd.DataFrame
            selected data in a dataframe with columns (mod_name,obs_name,x,y,mod_val,obs_val)

        See also
        --------
        skill 
            a method for aggregated skill assessment
        scatter
            a method for plotting compared data

        Examples
        --------
        >>> cc = mr.extract()        
        >>> dfsub = cc['c2'].sel_df(model=0)        
        >>> dfsub = cc['c2'].sel_df(start='2017-10-1', end='2017-11-1')
        >>> dfsub = cc['c2'].sel_df(area=[0.5,52.5,5,54])
        """
        # only for improved documentation
        return super().sel_df(
            model=model, observation=observation, start=start, end=end, area=area, df=df
        )

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

    def residual_hist(self, bins=100):
        plt.hist(self.residual, bins=bins, color=self._resi_color)
        plt.title(f"Residuals, {self.name}")
        plt.xlabel(f"Residuals of {self._obs_unit_text}")

    def hist(self, model=None, bins=100):
        """Plot histogram of model data and observations. 
        Wraps pandas.DataFrame hist() method.

        Parameters
        ----------
        model : (str, int), optional
            name or id of model to be plotted, by default None
        bins : int, optional
            number of bins, by default 100
        """
        mod_id = self._get_mod_id(model)
        mod_name = self.mod_names[mod_id]

        ax = self.df[mod_name].hist(
            bins=bins, color=self._mod_colors[mod_id], alpha=0.5
        )
        self.df[self.obs_name].hist(
            bins=bins, color=self.observation.color, alpha=0.5, ax=ax
        )
        ax.legend([mod_name, self.obs_name])
        plt.title(f"{mod_name} vs {self.name}")
        plt.xlabel(f"{self._obs_unit_text}")

    # def score(self, model=None, metric=None):

    #     mod_id = self._get_mod_id(model)

    #     if metric is None:
    #         metric = mtr.rmse

    #     return metric(self.obs, self.mod[:, mod_id])


class PointComparer(SingleObsComparer):
    """
    Comparer for observations from fixed locations

    Examples
    --------
    >>> mr = ModelResult("Oresund2D.dfsu")
    >>> o1 = PointObservation("klagshamn.dfs0", item=0, x=366844, y=6154291, name="Klagshamn")
    >>> mr.add_observation(o1, item=0)
    >>> comparer = mr.extract()
    >>> comparer['Klagshamn']
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
                self.mod_data[key].plot(ax=ax, color=self._mod_colors[j])

            ax.scatter(
                self.df.index,
                self.df[[self.obs_name]],
                marker=".",
                color=self.observation.color,
            )
            ax.set_ylabel(self._obs_unit_text)
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
                        line=dict(color=self._mod_colors[j]),
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

            fig.update_layout(title=title, yaxis_title=self._obs_unit_text, **kwargs)
            fig.update_yaxes(range=ylim)

            fig.show()
        else:
            raise ValueError(f"Plotting backend: {backend} not supported")


class TrackComparer(SingleObsComparer):
    """
    Comparer for observations from changing locations i.e. `TrackObservation`

    Examples
    --------
    >>> mr = ModelResult("HKZN_local_2017.dfsu")
    >>> c2 = TrackObservation("Alti_c2_Dutch.dfs0", item=3, name="c2")
    >>> mr.add_observation(c2, item=0)
    >>> comparer = mr.extract()
    >>> comparer['c2']
    """

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


class ComparerCollection(Mapping, BaseComparer):
    """
    Collection of comparers, constructed by calling the `ModelResult.extract` method.

    Examples
    --------
    >>> mr = ModelResult("Oresund2D.dfsu")
    >>> o1 = PointObservation("klagshamn.dfs0", item=0, x=366844, y=6154291, name="Klagshamn")
    >>> o2 = PointObservation("drogden.dfs0", item=0, x=355568.0, y=6156863.0)
    >>> mr.add_observation(o1, item=0)
    >>> mr.add_observation(o2, item=0)
    >>> comparer = mr.extract()

    """

    _all_df = None
    _start = datetime(2900, 1, 1)
    _end = datetime(1, 1, 1)
    _n_points = 0

    @property
    def name(self) -> str:
        return "Observations"

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
    def n_observations(self) -> int:
        return self.n_comparers

    @property
    def n_comparers(self) -> int:
        return len(self.comparers)

    def _construct_all_df(self):
        # TODO: var_name
        res = self._all_df_template()
        cols = res.keys()
        for cmp in self.comparers.values():
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
        self.comparers = {}
        self._mod_names = []
        self._obs_names = []

    def __repr__(self):
        out = []
        out.append(f"<{type(self).__name__}>")
        for key, value in self.comparers.items():
            out.append(f"{type(value).__name__}: {key}")
        return str.join("\n", out)

    def __getitem__(self, x):
        return self.comparers[self._get_obs_name(x)]

    def __len__(self) -> int:
        return len(self.comparers)

    def __iter__(self):
        return iter(self.comparers)

    def add_comparer(self, comparer: SingleObsComparer):
        """Add another Comparer to this collection.

        Parameters
        ----------
        comparer : (PointComparer, TrackComparer)
            Comparer to add to this collection
        """

        self.comparers[comparer.name] = comparer
        for mod_name in comparer.mod_names:
            if mod_name not in self._mod_names:
                self._mod_names.append(mod_name)
        self._obs_names.append(comparer.observation.name)
        self._n_points = self._n_points + comparer.n_points
        if comparer.start < self.start:
            self._start = comparer.start
        if comparer.end > self.end:
            self._end = comparer.end
        self._obs_unit_text = comparer.observation._unit_text()

        self._all_df = None

    def mean_skill(
        self,
        weights: Union[str, List[float]] = None,
        metrics: list = None,
        model: Union[str, int, List[str], List[int]] = None,
        observation: Union[str, int, List[str], List[int]] = None,
        start: Union[str, datetime] = None,
        end: Union[str, datetime] = None,
        area: List[float] = None,
        df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Weighted mean skill of model(s) over all observations        

        Parameters
        ----------
        weights : (str, List(float)), optional
            None: use assigned weights from observations
            "equal": giving all observations equal weight,
            "points": giving all points equal weight,
            list of weights e.g. [0.3, 0.3, 0.4] per observation, 
            by default None
        metrics : list, optional
            list of fmskill.metrics, by default [bias, rmse, urmse, mae, cc, si, r2]
        model : (str, int, List[str], List[int]), optional 
            name or ids of models to be compared, by default all
        observation : (str, int, List[str], List[int])), optional
            name or ids of observations to be compared, by default all
        start : (str, datetime), optional
            start time of comparison, by default None
        end : (str, datetime), optional
            end time of comparison, by default None
        area : list(float), optional
            bbox coordinates [x0, y0, x1, y1], 
            or polygon coordinates [x0, y0, x1, y1, ..., xn, yn], 
            by default None
        df : pd.dataframe, optional
            show user-provided data instead of the comparers own data, by default None

        Returns
        -------
        pd.DataFrame
            mean skill assessment as a dataframe

        See also
        --------
        skill
            a method for skill assessment observation by observation

        Examples
        --------
        >>> cc = mr.extract()
        >>> cc.mean_skill().round(2)
                    bias  rmse  urmse   mae    cc    si    r2
        HKZN_local -0.09  0.31   0.28  0.24  0.97  0.09  0.99
        """

        if metrics is None:
            metrics = [mtr.bias, mtr.rmse, mtr.urmse, mtr.mae, mtr.cc, mtr.si, mtr.r2]

        df = self.sel_df(
            df=df, model=model, observation=observation, start=start, end=end, area=area
        )
        mod_names = df.mod_name.unique()
        obs_names = df.obs_name.unique()
        n_obs = len(obs_names)
        n_metrics = len(metrics)

        weights = self._parse_weights(weights, observation)
        has_weights = False if (weights is None) else True

        rows = []
        for mod_name in mod_names:
            row = {}
            tmp = np.zeros((n_obs, n_metrics + 1))
            tmp_n = np.ones(n_obs, dtype=int)
            for obs_id, obs_name in enumerate(obs_names):
                dfsub = df[(df.mod_name == mod_name) & (df.obs_name == obs_name)]
                if len(dfsub) > 0:
                    tmp_n[obs_id] = len(dfsub)
                    for j, metric in enumerate(metrics):
                        tmp[obs_id, j] = metric(
                            dfsub.obs_val.values, dfsub.mod_val.values
                        )
            if not has_weights:
                weights = tmp_n

            weights = np.array(weights)
            tot_weight = np.sum(
                weights[tmp_n > 0]
            )  # this may be different for different models
            for j, metric in enumerate(metrics):
                row[metric.__name__] = np.inner(tmp[:, j], weights) / tot_weight
            rows.append(row)

        return pd.DataFrame(rows, index=mod_names)

    def _parse_weights(self, weights, observations):

        if observations is None:
            observations = self._obs_names
        else:
            observations = [observations] if np.isscalar(observations) else observations
            observations = [self._get_obs_name(o) for o in observations]
        n_obs = len(observations)

        if weights is None:
            # get weights from observation objects
            # default is equal weight to all
            weights = [self.comparers[o].observation.weight for o in observations]
        else:
            if isinstance(weights, int):
                weights = np.ones(n_obs)  # equal weight to all
            elif isinstance(weights, str):
                if weights.lower() == "equal":
                    weights = np.ones(n_obs)  # equal weight to all
                elif "points" in weights.lower():
                    weights = None  # no weight => use n_points
            elif not np.isscalar(weights):
                if not len(weights) == n_obs:
                    raise ValueError(
                        "weights must have length equal to number of observations"
                    )
        return weights

    def score(
        self,
        weights: Union[str, List[float]] = None,
        metric=mtr.rmse,
        model: Union[str, int, List[str], List[int]] = None,
        observation: Union[str, int, List[str], List[int]] = None,
        start: Union[str, datetime] = None,
        end: Union[str, datetime] = None,
        area: List[float] = None,
        df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Weighted mean score of model(s) over all observations        

        Parameters
        ----------
        weights : (str, List(float)), optional
            list of weights e.g. [0.3, 0.3, 0.4] per observation, 
            "equal": giving all observations equal weight,
            "points": giving all points equal weight,
            by default "equal"
        metric : list, optional
            a single metric from fmskill.metrics, by default rmse
        model : (str, int, List[str], List[int]), optional 
            name or ids of models to be compared, by default all
        observation : (str, int, List[str], List[int])), optional
            name or ids of observations to be compared, by default all
        start : (str, datetime), optional
            start time of comparison, by default None
        end : (str, datetime), optional
            end time of comparison, by default None
        area : list(float), optional
            bbox coordinates [x0, y0, x1, y1], 
            or polygon coordinates [x0, y0, x1, y1, ..., xn, yn], 
            by default None
        df : pd.dataframe, optional
            show user-provided data instead of the comparers own data, by default None

        Returns
        -------
        float
            mean skill score as a single number (for each model)

        See also
        --------
        skill
            a method for skill assessment observation by observation
        mean_skill
            a method for weighted mean skill assessment 

        Examples
        --------
        >>> cc = mr.extract()
        >>> cc.score()
        0.30681206
        >>> cc.score(weights=[0.1,0.1,0.8])
        0.3383011631797379
        
        >>> import fmskill.metrics as mtr
        >>> cc.score(weights='points', metric=mtr.mape)
        8.414442957854142
        """

        df = self.mean_skill(
            weights=weights,
            metrics=[metric],
            model=model,
            observation=observation,
            start=start,
            end=end,
            area=area,
            df=df,
        )
        values = df[metric.__name__].values
        if len(values) == 1:
            values = values[0]
        return values
