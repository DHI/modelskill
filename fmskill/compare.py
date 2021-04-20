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
from collections.abc import Mapping, Iterable
from typing import List, Union
import warnings
from inspect import getmembers, isfunction
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
from fmskill.plot import scatter


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
    def n_variables(self) -> int:
        return len(self._var_names)

    @property
    def all_df(self) -> pd.DataFrame:
        if self._all_df is None:
            self._construct_all_df()
        return self._all_df

    def _all_df_template(self):
        template = {
            "model": pd.Series([], dtype="category"),
            "observation": pd.Series([], dtype="category"),
        }
        if self.n_variables > 1:
            template["variable"] = pd.Series([], dtype="category")

        template["x"] = pd.Series([], dtype="float")
        template["y"] = pd.Series([], dtype="float")
        template["mod_val"] = pd.Series([], dtype="float")
        template["obs_val"] = pd.Series([], dtype="float")
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
            df["model"] = mod_name
            df["observation"] = self.observation.name
            if self.n_variables > 1:
                df["variable"] = self.observation.variable_name
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
        self._var_names = [observation.variable_name]
        self._itemInfos = [observation.itemInfo]

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
        out.append(f"Observation: {self.observation.name}, n_points={self.n_points}")
        for model in self.mod_names:
            out.append(f" Model: {model}, rmse={self.score(model=model):.3f}")
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

    def _get_var_name(self, var):
        return self._var_names[self._get_var_id(var)]

    def _get_var_id(self, var):
        if var is None or self.n_variables <= 1:
            return 0
        elif isinstance(var, str):
            if var in self._var_names:
                var_id = self._var_names.index(var)
            else:
                raise ValueError(f"var {var} could not be found in {self._var_names}")
        elif isinstance(var, int):
            if var >= 0 and var < self.n_variables:
                var_id = var
            else:
                raise ValueError(
                    f"var id was {var} - must be within 0 and {self.n_variables-1}"
                )
        else:
            raise ValueError("variable must be None, str or int")
        return var_id

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

    def _parse_metric(self, metric):
        if metric is None:
            return [mtr.bias, mtr.rmse, mtr.urmse, mtr.mae, mtr.cc, mtr.si, mtr.r2]

        if isinstance(metric, str):
            valid_metrics = [x[0] for x in getmembers(mtr, isfunction)]

            if metric.lower() in valid_metrics:
                metric = getattr(mtr, metric.lower())
            else:
                raise ValueError(
                    f"Invalid metric: {metric}. Valid metrics are {valid_metrics}."
                )
        elif isinstance(metric, Iterable):
            metrics = [self._parse_metric(m) for m in metric]
            return metrics
        elif not callable(metric):
            raise ValueError(
                f"Invalid metric: {metric}. Must be either string or callable."
            )
        return metric

    def skill(
        self,
        by: Union[str, List[str]] = None,
        metrics: list = None,
        model: Union[str, int, List[str], List[int]] = None,
        observation: Union[str, int, List[str], List[int]] = None,
        variable: Union[str, int, List[str], List[int]] = None,
        start: Union[str, datetime] = None,
        end: Union[str, datetime] = None,
        area: List[float] = None,
        df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Aggregated skill assessment of model(s)

        Parameters
        ----------
        by : (str, List[str]), optional
            group by column name or by temporal bin via the freq-argument
            (using pandas pd.Grouper(freq)),
            e.g.: 'freq:M' = monthly; 'freq:D' daily
            by default ["model","observation"]
        metrics : list, optional
            list of fmskill.metrics, by default [bias, rmse, urmse, mae, cc, si, r2]
        model : (str, int, List[str], List[int]), optional
            name or ids of models to be compared, by default all
        observation : (str, int, List[str], List[int])), optional
            name or ids of observations to be compared, by default all
        variable : (str, int, List[str], List[int])), optional
            name or ids of variables to be compared, by default all
        start : (str, datetime), optional
            start time of comparison, by default None
        end : (str, datetime), optional
            end time of comparison, by default None
        area : list(float), optional
            bbox coordinates [x0, y0, x1, y1],
            or polygon coordinates [x0, y0, x1, y1, ..., xn, yn],
            by default None
        df : pd.dataframe, optional
            user-provided data instead of the comparers own data, by default None

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

        >>> cc.skill(by='freq:D').round(2)
                      n  bias  rmse  urmse   mae    cc    si    r2
        2017-10-27  239 -0.15  0.25   0.21  0.20  0.72  0.10  0.98
        2017-10-28  162 -0.07  0.19   0.18  0.16  0.96  0.06  1.00
        2017-10-29  163 -0.21  0.52   0.47  0.42  0.79  0.11  0.99

        >>> df = cc.sel_df(observation=['HKNA','EPL']).copy()
        >>> df['seastate'] = pd.cut(df.obs_val, bins=[0,2,6], labels=['small','large'])
        >>> cc.skill(by=['observation','seastate'], df=df).round(2)
                                n  bias  rmse  urmse   mae    cc    si    r2
        observation seastate
        EPL         small      16  0.02  0.22   0.22  0.17  0.38  0.13  0.98
                    large      50 -0.11  0.22   0.19  0.19  0.98  0.06  0.99
        HKNA        small      61  0.02  0.09   0.09  0.08  0.88  0.05  1.00
                    large     324 -0.23  0.38   0.30  0.28  0.96  0.09  0.99
        """

        metrics = self._parse_metric(metrics)

        df = self.sel_df(
            model=model,
            observation=observation,
            variable=variable,
            start=start,
            end=end,
            area=area,
            df=df,
        )

        n_models = len(df.model.unique())
        n_obs = len(df.observation.unique())
        n_var = len(df.variable.unique()) if (self.n_variables > 1) else 1
        by = self._parse_by(by, n_models, n_obs, n_var)

        res = self._groupby_df(df.drop(columns=["x", "y"]), by, metrics)
        res = self._add_as_field_if_not_in_index(df, skilldf=res)
        return res

    def _add_as_field_if_not_in_index(
        self, df, skilldf, fields=["model", "observation", "variable"]
    ):
        """Add a field to skilldf if unique in df"""
        for field in reversed(fields):
            if (field == "model") and (self.n_models <= 1):
                continue
            if (field == "variable") and (self.n_variables <= 1):
                continue
            if field not in skilldf.index.names:
                unames = df[field].unique()
                if len(unames) == 1:
                    skilldf.insert(loc=0, column=field, value=unames[0])
        return skilldf

    def _groupby_df(self, df, by, metrics, n_min: int = None):
        def calc_metrics(x):
            row = {}
            row["n"] = len(x)
            for metric in metrics:
                row[metric.__name__] = metric(x.obs_val.values, x.mod_val.values)
            return pd.Series(row)

        # .drop(columns=["x", "y"])

        res = df.groupby(by=by).apply(calc_metrics)

        if n_min:
            # nan for all cols but n
            cols = [col for col in res.columns if not col == "n"]
            res.loc[res.n < n_min, cols] = np.nan

        res["n"] = res["n"].fillna(0)
        res = res.astype({"n": int})

        return res

    def _parse_by(self, by, n_models, n_obs, n_var=1):
        if by is None:
            by = []
            if n_models > 1:
                by.append("model")
            if n_obs > 1:  # or ((n_models == 1) and (n_obs == 1)):
                by.append("observation")
            if n_var > 1:
                by.append("variable")
            if len(by) == 0:
                # default value
                by.append("observation")
            return by

        if isinstance(by, str):
            if by in {"mdl", "mod", "models"}:
                by = "model"
            if by in {"obs", "observations"}:
                by = "observation"
            if by in {"var", "variables", "item"}:
                by = "variable"
            if by[:5] == "freq:":
                freq = by.split(":")[1]
                by = pd.Grouper(freq=freq)
        elif isinstance(by, Iterable):
            by = [self._parse_by(b, n_models, n_obs, n_var) for b in by]
            return by
        else:
            raise ValueError("Invalid by argument. Must be string or list of strings.")
        return by

    def spatial_skill(
        self,
        bins=5,
        binsize: float = None,
        by: Union[str, List[str]] = None,
        metrics: list = None,
        n_min: int = None,
        model: Union[str, int, List[str], List[int]] = None,
        observation: Union[str, int, List[str], List[int]] = None,
        variable: Union[str, int, List[str], List[int]] = None,
        start: Union[str, datetime] = None,
        end: Union[str, datetime] = None,
        area: List[float] = None,
        df: pd.DataFrame = None,
    ):
        """Aggregated spatial skill assessment of model(s) on a regular spatial grid.

        Parameters
        ----------
        bins: int, list of scalars, or IntervalIndex, or tuple of, optional
            criteria to bin x and y by, argument bins to pd.cut(), default 5
            define different bins for x and y a tuple
            e.g.: bins = 5, bins = (5,[2,3,5])
        binsize : float, optional
            bin size for x and y dimension, overwrites bins
            creates bins with reference to round(mean(x)), round(mean(y))
        by : (str, List[str]), optional
            group by column name or by temporal bin via the freq-argument
            (using pandas pd.Grouper(freq)),
            e.g.: 'freq:M' = monthly; 'freq:D' daily
            by default ["model","observation"]
        metrics : list, optional
            list of fmskill.metrics, by default [bias, rmse, urmse, mae, cc, si, r2]
        n_min : int, optional
            minimum number of observations in a grid cell;
            cells with fewer observations get a score of `np.nan`
        model : (str, int, List[str], List[int]), optional
            name or ids of models to be compared, by default all
        observation : (str, int, List[str], List[int])), optional
            name or ids of observations to be compared, by default all
        variable : (str, int, List[str], List[int])), optional
            name or ids of variables to be compared, by default all
        start : (str, datetime), optional
            start time of comparison, by default None
        end : (str, datetime), optional
            end time of comparison, by default None
        area : list(float), optional
            bbox coordinates [x0, y0, x1, y1],
            or polygon coordinates [x0, y0, x1, y1, ..., xn, yn],
            by default None
        df : pd.dataframe, optional
            user-provided data instead of the comparers own data, by default None

        Returns
        -------
        xr.Dataset
            skill assessment as a dataset

        See also
        --------
        skill
            a method for aggregated skill assessment

        Examples
        --------
        >>> cc = mr.extract()  # with satellite track measurements
        >>> cc.spatial_skill(metrics='bias')
        <xarray.Dataset>
        Dimensions:      (x: 5, y: 5)
        Coordinates:
            observation   'alti'
        * x            (x) float64 -0.436 1.543 3.517 5.492 7.466
        * y            (y) float64 50.6 51.66 52.7 53.75 54.8
        Data variables:
            n            (x, y) int32 3 0 0 14 37 17 50 36 72 ... 0 0 15 20 0 0 0 28 76
            bias         (x, y) float64 -0.02626 nan nan ... nan 0.06785 -0.1143

        >>> ds = cc.spatial_skill(binsize=0.5)
        >>> ds.coords
        Coordinates:
            observation   'alti'
        * x            (x) float64 -1.5 -0.5 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5
        * y            (y) float64 51.5 52.5 53.5 54.5 55.5 56.5
        """

        metrics = self._parse_metric(metrics)

        df = self.sel_df(
            model=model,
            observation=observation,
            variable=variable,
            start=start,
            end=end,
            area=area,
            df=df,
        )

        df = self._add_spatial_grid_to_df(df=df, bins=bins, binsize=binsize)

        n_models = len(df.model.unique())
        n_obs = len(df.observation.unique())
        by = self._parse_by(by, n_models, n_obs)
        if not "y" in by:
            by.insert(0, "y")
        if not "x" in by:
            by.insert(0, "x")

        res = self._groupby_df(
            df.drop(columns=["x", "y"]).rename(columns=dict(xBin="x", yBin="y")),
            by,
            metrics,
            n_min,
        )

        return res.to_xarray().squeeze()

    def _add_spatial_grid_to_df(self, df, bins, binsize):
        if binsize is None:
            # bins from bins
            if isinstance(bins, tuple):
                bins_x = bins[0]
                bins_y = bins[1]
            else:
                bins_x = bins
                bins_y = bins
        else:
            # bins from binsize
            x_ptp = df.x.values.ptp()
            y_ptp = df.y.values.ptp()
            nx = int(np.ceil(x_ptp / binsize))
            ny = int(np.ceil(y_ptp / binsize))
            x_mean = np.round(df.x.mean())
            y_mean = np.round(df.y.mean())
            bins_x = np.arange(
                x_mean - nx / 2 * binsize, x_mean + (nx / 2 + 1) * binsize, binsize
            )
            bins_y = np.arange(
                y_mean - ny / 2 * binsize, y_mean + (ny / 2 + 1) * binsize, binsize
            )
        # cut and get bin centre
        df["xBin"] = pd.cut(df.x, bins=bins_x)
        df["xBin"] = df["xBin"].apply(lambda x: x.mid)
        df["yBin"] = pd.cut(df.y, bins=bins_y)
        df["yBin"] = df["yBin"].apply(lambda x: x.mid)

        return df

    def sel_df(
        self,
        model: Union[str, int, List[str], List[int]] = None,
        observation: Union[str, int, List[str], List[int]] = None,
        variable: Union[str, int, List[str], List[int]] = None,
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
        variable : (str, int, List[str], List[int])), optional
            name or ids of variables to be compared, by default all
        start : (str, datetime), optional
            start time of comparison, by default None
        end : (str, datetime), optional
            end time of comparison, by default None
        area : list(float), optional
            bbox coordinates [x0, y0, x1, y1],
            or polygon coordinates [x0, y0, x1, y1, ..., xn, yn],
            by default None
        df : pd.dataframe, optional
            user-provided data instead of the comparers own data, by default None

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
                           model observation      x       y   mod_val  obs_val
        2017-10-28 01:00:00 SW_1         EPL  3.276  51.999  1.644092     1.82
        2017-10-28 02:00:00 SW_1         EPL  3.276  51.999  1.755809     1.86
        2017-10-28 03:00:00 SW_1         EPL  3.276  51.999  1.867526     2.11
        """
        if df is None:
            df = self.all_df
        if model is not None:
            models = [model] if np.isscalar(model) else model
            models = [self._get_mod_name(m) for m in models]
            df = df[df.model.isin(models)]
        if observation is not None:
            observation = [observation] if np.isscalar(observation) else observation
            observation = [self._get_obs_name(o) for o in observation]
            df = df[df.observation.isin(observation)]
        if (variable is not None) and (self.n_variables > 1):
            variable = [variable] if np.isscalar(variable) else variable
            variable = [self._get_var_name(v) for v in variable]
            df = df[df.variable.isin(variable)]
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

    def _area_is_polygon(self, area) -> bool:
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
            if len(polygon) <= 5:
                return False
            if len(polygon) % 2 != 0:
                return False

        if polygon.ndim == 2:
            if polygon.shape[0] < 3:
                return False
            if polygon.shape[1] != 2:
                return False

        return True

    def _inside_polygon(self, polygon, xy):
        import matplotlib.path as mp

        if polygon.ndim == 1:
            polygon = np.column_stack((polygon[0::2], polygon[1::2]))
        return mp.Path(polygon).contains_points(xy)

    def scatter(
        self,
        *,
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
        variable: Union[str, int, List[str], List[int]] = None,
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
            name or id of model to be compared, by default first
        observation : (int, str, List[str], List[int])), optional
            name or ids of observations to be compared, by default None
        variable : (str, int), optional
            name or id of variable to be compared, by default first
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
        # select model
        mod_id = self._get_mod_id(model)
        mod_name = self.mod_names[mod_id]

        # select variable
        var_id = self._get_var_id(variable)
        var_name = self._var_names[var_id]

        # filter data
        df = self.sel_df(
            df=df,
            model=mod_name,
            observation=observation,
            variable=var_name,
            start=start,
            end=end,
            area=area,
        )
        if len(df) == 0:
            raise Exception("No data found in selection")

        x = df.obs_val
        y = df.mod_val

        unit_text = self._obs_unit_text
        if isinstance(self, ComparerCollection):
            unit_text = self[df.observation[0]]._obs_unit_text

        if xlabel is None:
            xlabel = f"Observation, {unit_text}"

        if ylabel is None:
            ylabel = f"Model, {unit_text}"

        if title is None:
            title = f"{self.mod_names[mod_id]} vs {self.name}"

        if show_points is None:
            show_points = len(x) < 1e4

        scatter(
            x=x,
            y=y,
            binsize=binsize,
            nbins=nbins,
            show_points=show_points,
            show_hist=show_hist,
            backend=backend,
            figsize=figsize,
            xlim=xlim,
            ylim=ylim,
            reg_method=reg_method,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            **kwargs,
        )


class SingleObsComparer(BaseComparer):
    def __add__(self, other):
        cc = ComparerCollection()
        cc.add_comparer(self)
        if isinstance(other, SingleObsComparer):
            cc.add_comparer(other)
        elif isinstance(other, ComparerCollection):
            for c in other:
                cc.add_comparer(c)
        else:
            raise TypeError(f"Cannot add {type(other)} to {type(self)}")
        return cc

    def __copy__(self):
        # cls = self.__class__
        # cp = cls.__new__(cls)
        # cp.__init__(self.observation, self.mod_df)
        # return cp
        return deepcopy(self)

    def copy(self):
        return self.__copy__()

    def skill(
        self,
        by: Union[str, List[str]] = None,
        metrics: list = None,
        model: Union[str, int, List[str], List[int]] = None,
        start: Union[str, datetime] = None,
        end: Union[str, datetime] = None,
        area: List[float] = None,
        df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Skill assessment of model(s)

        Parameters
        ----------
        by : (str, List[str]), optional
            group by column name or by temporal bin via the freq-argument
            (using pandas pd.Grouper(freq)),
            e.g.: 'freq:M' = monthly; 'freq:D' daily
            by default ["model"]
        metrics : list, optional
            list of fmskill.metrics, by default [bias, rmse, urmse, mae, cc, si, r2]
        model : (str, int, List[str], List[int]), optional
            name or ids of models to be compared, by default all
        freq : string, optional
            do temporal binning using pandas pd.Grouper(freq),
            typical examples: 'M' = monthly; 'D' daily
            by default None
        start : (str, datetime), optional
            start time of comparison, by default None
        end : (str, datetime), optional
            end time of comparison, by default None
        area : list(float), optional
            bbox coordinates [x0, y0, x1, y1],
            or polygon coordinates [x0, y0, x1, y1, ..., xn, yn],
            by default None
        df : pd.dataframe, optional
            user-provided data instead of the comparers own data, by default None

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

        >>> cc['c2'].skill(by='freq:D').round(2)
                     n  bias  rmse  urmse   mae    cc    si    r2
        2017-10-27  72 -0.19  0.31   0.25  0.26  0.48  0.12  0.98
        2017-10-28   0   NaN   NaN    NaN   NaN   NaN   NaN   NaN
        2017-10-29  41  0.33  0.41   0.25  0.36  0.96  0.06  0.99

        >>> df = cc['c2'].sel_df().copy()
        >>> df['Hm0 group'] = pd.cut(df.obs_val, bins=[0,2,6])
        >>> cc['c2'].skill(by='Hm0 group', df=df).round(2)
                    n  bias  rmse  urmse   mae    cc    si    r2
        Hm0 group
        (0, 2]     33 -0.09  0.23   0.22  0.21  0.46  0.12  0.98
        (2, 6]     80  0.03  0.39   0.39  0.33  0.97  0.12  0.99
        """
        # only for improved documentation
        return super().skill(
            model=model,
            by=by,
            start=start,
            end=end,
            area=area,
            df=df,
            metrics=metrics,
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
            user-provided data instead of the comparers own data, by default None

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
        metric = self._parse_metric(metric)

        df = self.skill(
            metrics=[metric],
            model=model,
            start=start,
            end=end,
            area=area,
            df=df,
        )
        values = df[metric.__name__].values
        if len(values) == 1:
            values = values[0]
        return values

    def sel_df(
        self,
        model: Union[str, int, List[str], List[int]] = None,
        observation: Union[str, int, List[str], List[int]] = None,
        variable: Union[str, int, List[str], List[int]] = None,
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
            user-provided data instead of the comparers own data, by default None

        Returns
        -------
        pd.DataFrame
            selected data in a dataframe with columns (model,observation,x,y,mod_val,obs_val)

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
            model=model,
            observation=observation,
            variable=variable,
            start=start,
            end=end,
            area=area,
            df=df,
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

        elif backend == "plotly":  # pragma: no cover
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
    def var_names(self):
        """List of variable names"""
        return self._var_names

    @var_names.setter
    def var_names(self, value):
        if np.isscalar(value):
            value = [value]
        if len(value) != self.n_variables:
            raise ValueError(f"Length of var_names must be {self.n_variables}")
        for var_id, new_var in enumerate(value):
            for c in self.comparers.values():
                if c._var_names[0] == self.var_names[var_id]:
                    c.observation.variable_name = new_var
                    c._var_names = [new_var]
        if self.n_variables > 1:
            if self._all_df is not None:
                self._all_df["variable"]
                for old_var, new_var in zip(self.var_names, value):
                    self._all_df.loc[
                        self._all_df.variable == old_var, "variable"
                    ] = new_var
        self._var_names = value

    @property
    def obs_names(self):
        """List of observation names"""
        return self._var_names

    @property
    def n_observations(self) -> int:
        """Number of observations"""
        return self.n_comparers

    @property
    def n_comparers(self) -> int:
        """Number of comparers"""
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
                df["model"] = mod_name
                df["observation"] = cmp.observation.name
                if self.n_variables > 1:
                    df["variable"] = cmp.observation.variable_name
                df["x"] = cmp.x
                df["y"] = cmp.y
                df["obs_val"] = cmp.obs
                res = res.append(df[cols])

        self._all_df = res.sort_index()

    def __init__(self):
        self.comparers = {}
        self._mod_names = []
        self._obs_names = []
        self._var_names = []
        self._itemInfos = []

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

    def __add__(self, other):
        # if type(other) not in (SingleObsComparer, ComparerCollection):
        #    raise TypeError(f"Cannot add {type(other)} to ComparerCollection")

        cp = self.copy()
        if isinstance(other, SingleObsComparer):
            cp.add_comparer(other)
        elif isinstance(other, ComparerCollection):
            for c in other:
                cp.add_comparer(c)
        return cp

    def __copy__(self):
        cls = self.__class__
        cp = cls.__new__(cls)
        cp.__init__()
        for c in self.comparers.values():
            cp.add_comparer(c)
        return cp

    def copy(self):
        return self.__copy__()

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
        if comparer.observation.variable_name not in self._var_names:
            self._var_names.append(comparer.observation.variable_name)

        # check if already in...
        self._itemInfos.append(comparer.observation.itemInfo)

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
        variable: Union[str, int, List[str], List[int]] = None,
        start: Union[str, datetime] = None,
        end: Union[str, datetime] = None,
        area: List[float] = None,
        df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Weighted mean skill of model(s) over all observations (of same variable)

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
        variable : (str, int, List[str], List[int])), optional
            name or ids of variables to be compared, by default all
        start : (str, datetime), optional
            start time of comparison, by default None
        end : (str, datetime), optional
            end time of comparison, by default None
        area : list(float), optional
            bbox coordinates [x0, y0, x1, y1],
            or polygon coordinates [x0, y0, x1, y1, ..., xn, yn],
            by default None
        df : pd.dataframe, optional
            user-provided data instead of the comparers own data, by default None

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
                      n  bias  rmse  urmse   mae    cc    si    r2
        HKZN_local  564 -0.09  0.31   0.28  0.24  0.97  0.09  0.99
        """
        # TODO: how to handle by=freq:D?

        # filter data
        df = self.sel_df(
            df=df,
            model=model,
            observation=observation,
            variable=variable,
            start=start,
            end=end,
            area=area,
        )
        mod_names = df.model.unique()
        obs_names = df.observation.unique()
        var_names = self.var_names
        if self.n_variables > 1:
            var_names = df.variable.unique()
        n_models = len(mod_names)

        # skill assessment
        metrics = self._parse_metric(metrics)
        skilldf = self.skill(df=df, metrics=metrics)

        # weights
        weights = self._parse_weights(weights, obs_names)
        skilldf["weights"] = (
            skilldf.n if weights is None else np.repeat(weights, n_models)
        )
        weighted_mean = lambda x: np.average(x, weights=skilldf.loc[x.index, "weights"])

        # group by
        by = self._mean_skill_by(skilldf, mod_names, var_names)
        agg = {"n": np.sum}
        for metric in metrics:
            agg[metric.__name__] = weighted_mean
        res = skilldf.groupby(by).agg(agg)

        # output
        res = self._add_as_field_if_not_in_index(df, res, fields=["model", "variable"])
        return res.astype({"n": int})

    def _mean_skill_by(self, skilldf, mod_names, var_names):
        by = []
        if len(mod_names) > 1:
            by.append("model")
        if len(var_names) > 1:
            by.append("variable")
        if len(by) == 0:
            if (self.n_variables > 1) and ("variable" in skilldf):
                by.append("variable")
            elif "model" in skilldf:
                by.append("model")
            else:
                by = [mod_names[0]] * len(skilldf)
        return by

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
                elif "point" in weights.lower():
                    weights = None  # no weight => use n_points
                else:
                    raise ValueError(
                        "unknown weights argument (None, 'equal', 'points', or list of floats)"
                    )
            elif not np.isscalar(weights):
                if n_obs == 1:
                    if len(weights) > 1:
                        warnings.warn(
                            "Cannot apply multiple weights to one observation"
                        )
                    weights = [1.0]
                if not len(weights) == n_obs:
                    raise ValueError(
                        f"weights must have same length as observations: {observations}"
                    )
        return weights

    def score(
        self,
        weights: Union[str, List[float]] = None,
        metric=mtr.rmse,
        model: Union[str, int, List[str], List[int]] = None,
        observation: Union[str, int, List[str], List[int]] = None,
        variable: Union[str, int, List[str], List[int]] = None,
        start: Union[str, datetime] = None,
        end: Union[str, datetime] = None,
        area: List[float] = None,
        df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Weighted mean score of model(s) over all observations
        NOTE: will take simple mean over different variables

        Parameters
        ----------
        weights : (str, List(float)), optional
            None: use assigned weights from observations
            "equal": giving all observations equal weight,
            "points": giving all points equal weight,
            list of weights e.g. [0.3, 0.3, 0.4] per observation,
            by default None
        metric : list, optional
            a single metric from fmskill.metrics, by default rmse
        model : (str, int, List[str], List[int]), optional
            name or ids of models to be compared, by default all
        observation : (str, int, List[str], List[int])), optional
            name or ids of observations to be compared, by default all
        variable : (str, int, List[str], List[int])), optional
            name or ids of variables to be compared, by default all
        start : (str, datetime), optional
            start time of comparison, by default None
        end : (str, datetime), optional
            end time of comparison, by default None
        area : list(float), optional
            bbox coordinates [x0, y0, x1, y1],
            or polygon coordinates [x0, y0, x1, y1, ..., xn, yn],
            by default None
        df : pd.dataframe, optional
            user-provided data instead of the comparers own data, by default None

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

        >>> cc.score(weights='points', metric="mape")
        8.414442957854142
        """
        metric = self._parse_metric(metric)

        if model is None:
            models = self._mod_names
        else:
            models = [model] if np.isscalar(model) else model
            models = [self._get_mod_name(m) for m in models]
        n_models = len(models)

        df = self.mean_skill(
            weights=weights,
            metrics=[metric],
            model=models,
            observation=observation,
            variable=variable,
            start=start,
            end=end,
            area=area,
            df=df,
        )

        if n_models == 1:
            score = df[metric.__name__].values.mean()
        else:
            score = {}
            for model in models:
                mtr_val = df.loc[model][metric.__name__]
                if not np.isscalar(mtr_val):
                    # e.g. mean over different variables!
                    mtr_val = mtr_val.values.mean()
                score[model] = mtr_val

        return score
