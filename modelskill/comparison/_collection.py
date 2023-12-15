from __future__ import annotations
import os
from pathlib import Path
import tempfile
from typing import (
    Callable,
    Dict,
    List,
    Union,
    Optional,
    Mapping,
    Iterable,
    overload,
    Hashable,
)
import warnings
import zipfile
import numpy as np
import pandas as pd


from .. import metrics as mtr
from ..plotting import taylor_diagram, TaylorPoint

from ._collection_plotter import ComparerCollectionPlotter
from ..skill import SkillTable
from ..skill_grid import SkillGrid
from ..settings import options, reset_option

from ..utils import _get_idx, _get_name
from ._comparison import Comparer, Scoreable
from ._utils import (
    _parse_metric,
    _add_spatial_grid_to_df,
    _groupby_df,
    _parse_groupby,
    IdOrNameTypes,
    TimeTypes,
)
from ._comparison import _get_deprecated_args  # TODO remove in v 1.1


def _get_deprecated_obs_var_args(kwargs):
    observation, variable = None, None

    # Don't bother refactoring this, it will be removed in v1.1
    if "observation" in kwargs:
        observation = kwargs.pop("observation")
        if observation is not None:
            warnings.warn(
                f"The 'observation' argument is deprecated, use 'sel(observation='{observation}') instead",
                FutureWarning,
            )

    if "variable" in kwargs:
        variable = kwargs.pop("variable")

        if variable is not None:
            warnings.warn(
                f"The 'variable' argument is deprecated, use 'sel(variable='{variable}') instead",
                FutureWarning,
            )

    return observation, variable


def _all_df_template(n_variables: int = 1):
    template = {
        "model": pd.Series([], dtype="category"),
        "observation": pd.Series([], dtype="category"),
    }
    if n_variables > 1:
        template["variable"] = pd.Series([], dtype="category")

    template["x"] = pd.Series([], dtype="float")
    template["y"] = pd.Series([], dtype="float")
    template["mod_val"] = pd.Series([], dtype="float")
    template["obs_val"] = pd.Series([], dtype="float")
    res = pd.DataFrame(template)
    return res


class ComparerCollection(Mapping, Scoreable):
    """
    Collection of comparers, constructed by calling the `modelskill.match` method.

    Examples
    --------
    >>> import modelskill as ms
    >>> mr = ms.ModelResult("Oresund2D.dfsu", item=0)
    >>> o1 = ms.PointObservation("klagshamn.dfs0", item=0, x=366844, y=6154291, name="Klagshamn")
    >>> o2 = ms.PointObservation("drogden.dfs0", item=0, x=355568.0, y=6156863.0)
    >>> cc = ms.match(obs=[o1,o2], mod=mr)
    """

    plotter = ComparerCollectionPlotter

    """Collection of Comparers, indexed by name"""

    def __init__(self, comparers: Iterable[Comparer]) -> None:
        self.comparers: Dict[str, Comparer] = {}
        self._insert_comparers(comparers)
        self.plot = ComparerCollection.plotter(self)

    def _insert_comparers(self, comparer: Union[Comparer, Iterable[Comparer]]) -> None:
        if isinstance(comparer, Iterable):
            for c in comparer:
                self[c.name] = c
        elif isinstance(comparer, Comparer):
            self[comparer.name] = comparer
        else:
            pass

    @property
    def name(self) -> str:
        return "Observations"

    @property
    def unit_text(self) -> str:
        # Picking the first one is arbitrary, but it should be the same for all
        # we could check that they are all the same, but let's assume that they are
        # for cmp in self:
        #     if cmp.unit_text != text:
        #         warnings.warn(f"Unit text is inconsistent: {text} vs {cmp.unit_text}")
        return self[0].unit_text

    @property
    def n_comparers(self) -> int:
        """Number of comparers"""
        return len(self.comparers)

    @property
    def n_points(self) -> int:
        """number of compared points"""
        return sum([c.n_points for c in self.comparers.values()])

    @property
    def start(self) -> pd.Timestamp:
        """start timestamp of compared data"""
        starts = [pd.Timestamp.max]
        for cmp in self.comparers.values():
            starts.append(cmp.time[0])
        return min(starts)

    @property
    def end(self) -> pd.Timestamp:
        """end timestamp of compared data"""
        ends = [pd.Timestamp.min]
        for cmp in self.comparers.values():
            ends.append(cmp.time[-1])
        return max(ends)

    @property
    def obs_names(self) -> List[str]:
        """List of observation names"""
        return [c.name for c in self.comparers.values()]

    @property
    def n_observations(self) -> int:
        """Number of observations"""
        return self.n_comparers

    @property
    def mod_names(self) -> List[str]:
        """List of unique model names"""
        unique_names = []
        for cmp in self.comparers.values():
            for n in cmp.mod_names:
                if n not in unique_names:
                    unique_names.append(n)
        return unique_names

    @property
    def n_models(self) -> int:
        return len(self.mod_names)

    @property
    def var_names(self) -> List[str]:
        """List of unique variable names"""
        unique_names = []
        for cmp in self.comparers.values():
            n = cmp.quantity.name
            if n not in unique_names:
                unique_names.append(n)
        return unique_names

    @property
    def n_variables(self) -> int:
        return len(self.var_names)

    @property
    def metrics(self):
        return options.metrics.list

    @metrics.setter
    def metrics(self, values) -> None:
        if values is None:
            reset_option("metrics.list")
        else:
            options.metrics.list = _parse_metric(values, self.metrics)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the data as a pandas DataFrame"""
        # TODO: var_name
        # TODO delegate to each comparer
        res = _all_df_template(self.n_variables)
        frames = []
        cols = res.keys()
        for cmp in self.comparers.values():
            for j in range(cmp.n_models):
                mod_name = cmp.mod_names[j]
                # drop "x", "y",  ?
                df = cmp.data.drop_vars(["z"])[[mod_name]].to_dataframe().copy()
                df = df.rename(columns={mod_name: "mod_val"})
                df["model"] = mod_name
                df["observation"] = cmp.name
                if self.n_variables > 1:
                    df["variable"] = cmp.quantity.name
                df["x"] = cmp.x
                df["y"] = cmp.y
                df["obs_val"] = cmp.obs
                frames.append(df[cols])
        if len(frames) > 0:
            res = pd.concat(frames)
        res = res.sort_index()
        res.index.name = "time"
        return res

    def __repr__(self):
        out = []
        out.append(f"<{type(self).__name__}>")
        for key, value in self.comparers.items():
            out.append(f"{type(value).__name__}: {key}")
        return str.join("\n", out)

    @overload
    def __getitem__(self, x: slice | Iterable[Hashable]) -> ComparerCollection:
        ...

    @overload
    def __getitem__(self, x: int | Hashable) -> Comparer:
        ...

    def __getitem__(self, x):
        if isinstance(x, str):
            return self.comparers[x]

        if isinstance(x, slice):
            idxs = list(range(*x.indices(len(self))))
            return ComparerCollection([self[i] for i in idxs])

        if isinstance(x, int):
            name = _get_name(x, self.obs_names)
            return self.comparers[name]

        if isinstance(x, Iterable):
            cmps = [self[i] for i in x]
            return ComparerCollection(cmps)

    def __setitem__(self, x: str, value: Comparer) -> None:
        assert isinstance(
            value, Comparer
        ), f"comparer must be a Comparer, not {type(value)}"
        if x in self.comparers:
            # comparer with this name already exists!
            # maybe the user is trying to add a new model
            # or a new time period
            self.comparers[x] = self.comparers[x] + value  # type: ignore
        else:
            self.comparers[x] = value

    def __len__(self) -> int:
        return len(self.comparers)

    def __iter__(self):
        return iter(self.comparers.values())

    def __copy__(self):
        cls = self.__class__
        cp = cls.__new__(cls)
        cp.__init__(list(self.comparers))  # TODO should this use deepcopy?
        return cp

    def copy(self):
        return self.__copy__()

    def __add__(
        self, other: Union["Comparer", "ComparerCollection"]
    ) -> "ComparerCollection":
        if not isinstance(other, (Comparer, ComparerCollection)):
            raise TypeError(f"Cannot add {type(other)} to {type(self)}")

        if isinstance(other, Comparer):
            return ComparerCollection([*self, other])
        elif isinstance(other, ComparerCollection):
            return ComparerCollection([*self, *other])

    def sel(
        self,
        model: Optional[IdOrNameTypes] = None,
        observation: Optional[IdOrNameTypes] = None,
        variable: Optional[IdOrNameTypes] = None,
        start: Optional[TimeTypes] = None,
        end: Optional[TimeTypes] = None,
        time: Optional[TimeTypes] = None,
        area: Optional[List[float]] = None,
        **kwargs,
    ) -> "ComparerCollection":
        """Select data based on model, time and/or area.

        Parameters
        ----------
        model : str or int or list of str or list of int, optional
            Model name or index. If None, all models are selected.
        observation : str or int or list of str or list of int, optional
            Observation name or index. If None, all observations are selected.
        variable : str or int or list of str or list of int, optional
            Variable name or index. If None, all variables are selected.
        start : str or datetime, optional
            Start time. If None, all times are selected.
        end : str or datetime, optional
            End time. If None, all times are selected.
        time : str or datetime, optional
            Time. If None, all times are selected.
        area : list of float, optional
            bbox: [x0, y0, x1, y1] or Polygon. If None, all areas are selected.
        kwargs : dict, optional
            Filtering by comparer attrs similar to xarray.Dataset.filter_by_attrs
            e.g. `sel(gtype='track')` or `sel(obs_provider='CMEMS')` if at least
            one comparer has an entry `obs_provider` with value `CMEMS` in its
            attrs container. Multiple kwargs are combined with logical AND.

        Returns
        -------
        ComparerCollection
            New ComparerCollection with selected data.
        """
        # TODO is this really necessary to do both in ComparerCollection and Comparer?
        if model is not None:
            if isinstance(model, (str, int)):
                models = [model]
            else:
                models = list(model)
            mod_names: List[str] = [_get_name(m, self.mod_names) for m in models]
        if observation is None:
            observation = self.obs_names
        else:
            observation = [observation] if np.isscalar(observation) else observation  # type: ignore
            observation = [_get_name(o, self.obs_names) for o in observation]  # type: ignore

        if (variable is not None) and (self.n_variables > 1):
            variable = [variable] if np.isscalar(variable) else variable  # type: ignore
            variable = [_get_name(v, self.var_names) for v in variable]  # type: ignore
        else:
            variable = self.var_names

        cmps = []
        for cmp in self.comparers.values():
            if cmp.name in observation and cmp.quantity.name in variable:
                thismodel = (
                    [m for m in mod_names if m in cmp.mod_names] if model else None
                )
                if (thismodel is not None) and (len(thismodel) == 0):
                    continue
                cmpsel = cmp.sel(
                    model=thismodel,
                    start=start,
                    end=end,
                    time=time,
                    area=area,
                )
                if cmpsel is not None:
                    # TODO: check if cmpsel is empty
                    if cmpsel.n_points > 0:
                        cmps.append(cmpsel)
        cc = ComparerCollection(cmps)

        if kwargs:
            cc = cc.filter_by_attrs(**kwargs)

        return cc

    def filter_by_attrs(self, **kwargs) -> "ComparerCollection":
        """Filter by comparer attrs similar to xarray.Dataset.filter_by_attrs

        Parameters
        ----------
        kwargs : dict, optional
            Filtering by comparer attrs similar to xarray.Dataset.filter_by_attrs
            e.g. `sel(gtype='track')` or `sel(obs_provider='CMEMS')` if at least
            one comparer has an entry `obs_provider` with value `CMEMS` in its
            attrs container. Multiple kwargs are combined with logical AND.

        Returns
        -------
        ComparerCollection
            New ComparerCollection with selected data.

        Examples
        --------
        >>> cc = ms.match([HKNA, EPL, alti], mr)
        >>> cc.filter_by_attrs(gtype='track')
        <ComparerCollection>
        Comparer: alti
        """
        cmps = []
        for cmp in self.comparers.values():
            for k, v in kwargs.items():
                # TODO: should we also filter on cmp.data.Observation.attrs?
                if cmp.data.attrs.get(k) != v:
                    break
            else:
                cmps.append(cmp)
        return ComparerCollection(cmps)

    def query(self, query: str) -> "ComparerCollection":
        """Select data based on a query.

        Parameters
        ----------
        query : str
            Query string. See pandas.DataFrame.query() for details.

        Returns
        -------
        ComparerCollection
            New ComparerCollection with selected data.
        """
        q_cmps = [cmp.query(query) for cmp in self.comparers.values()]
        cmps_with_data = [cmp for cmp in q_cmps if cmp.n_points > 0]

        return ComparerCollection(cmps_with_data)

    def skill(
        self,
        by: Optional[Union[str, List[str]]] = None,
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> SkillTable:
        """Aggregated skill assessment of model(s)

        Parameters
        ----------
        by : (str, List[str]), optional
            group by column name or by temporal bin via the freq-argument
            (using pandas pd.Grouper(freq)),
            e.g.: 'freq:M' = monthly; 'freq:D' daily
            by default ["model","observation"]
        metrics : list, optional
            list of modelskill.metrics, by default modelskill.options.metrics.list

        Returns
        -------
        pd.DataFrame
            skill assessment as a dataframe

        See also
        --------
        sel
            a method for filtering/selecting data

        Examples
        --------
        >>> import modelskill as ms
        >>> cc = ms.match([HKNA,EPL,c2], mr)
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
        """
        metrics = _parse_metric(metrics, self.metrics, return_list=True)

        # TODO remove in v1.1
        model, start, end, area = _get_deprecated_args(kwargs)
        observation, variable = _get_deprecated_obs_var_args(kwargs)
        assert kwargs == {}, f"Unknown keyword arguments: {kwargs}"

        cmp = self.sel(
            model=model,
            observation=observation,
            variable=variable,
            start=start,
            end=end,
            area=area,
        )
        if cmp.n_points == 0:
            raise ValueError("Dataset is empty, no data to compare.")

        ## ---- end of deprecated code ----

        df = cmp.to_dataframe()
        n_models = cmp.n_models  # len(df.model.unique())
        n_obs = cmp.n_observations  # len(df.observation.unique())

        # TODO: FIX
        n_var = (
            cmp.n_variables
        )  # len(df.variable.unique()) if (self.n_variables > 1) else 1
        by = _parse_groupby(by, n_models, n_obs, n_var)

        res = _groupby_df(df.drop(columns=["x", "y"]), by, metrics)
        res = cmp._add_as_col_if_not_in_index(df, skilldf=res)
        return SkillTable(res)

    def _add_as_col_if_not_in_index(
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

    def spatial_skill(
        self,
        bins=5,
        binsize=None,
        by=None,
        metrics=None,
        n_min=None,
        **kwargs,
    ):
        warnings.warn(
            "spatial_skill is deprecated, use gridded_skill instead", FutureWarning
        )
        return self.gridded_skill(
            bins=bins,
            binsize=binsize,
            by=by,
            metrics=metrics,
            n_min=n_min,
            **kwargs,
        )

    def gridded_skill(
        self,
        bins=5,
        binsize: Optional[float] = None,
        by: Optional[Union[str, List[str]]] = None,
        metrics: Optional[list] = None,
        n_min: Optional[int] = None,
        **kwargs,
    ) -> SkillGrid:
        """Skill assessment of model(s) on a regular spatial grid.

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
            list of modelskill.metrics, by default modelskill.options.metrics.list
        n_min : int, optional
            minimum number of observations in a grid cell;
            cells with fewer observations get a score of `np.nan`

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
        >>> import modelskill as ms
        >>> cc = ms.match([HKNA,EPL,c2], mr)  # with satellite track measurements
        >>> cc.gridded_skill(metrics='bias')
        <xarray.Dataset>
        Dimensions:      (x: 5, y: 5)
        Coordinates:
            observation   'alti'
        * x            (x) float64 -0.436 1.543 3.517 5.492 7.466
        * y            (y) float64 50.6 51.66 52.7 53.75 54.8
        Data variables:
            n            (x, y) int32 3 0 0 14 37 17 50 36 72 ... 0 0 15 20 0 0 0 28 76
            bias         (x, y) float64 -0.02626 nan nan ... nan 0.06785 -0.1143

        >>> ds = cc.gridded_skill(binsize=0.5)
        >>> ds.coords
        Coordinates:
            observation   'alti'
        * x            (x) float64 -1.5 -0.5 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5
        * y            (y) float64 51.5 52.5 53.5 54.5 55.5 56.5
        """

        model, start, end, area = _get_deprecated_args(kwargs)
        observation, variable = _get_deprecated_obs_var_args(kwargs)
        assert kwargs == {}, f"Unknown keyword arguments: {kwargs}"

        metrics = _parse_metric(metrics, self.metrics, return_list=True)

        cmp = self.sel(
            model=model,
            observation=observation,
            variable=variable,
            start=start,
            end=end,
            area=area,
        )

        if cmp.n_points == 0:
            raise ValueError("Dataset is empty, no data to compare.")

        ## ---- end of deprecated code ----

        df = cmp.to_dataframe()
        df = _add_spatial_grid_to_df(df=df, bins=bins, binsize=binsize)

        by = _parse_groupby(by, cmp.n_models, cmp.n_observations)
        if isinstance(by, str) or (not isinstance(by, Iterable)):
            by = [by]  # type: ignore
        if "x" not in by:  # type: ignore
            by.insert(0, "x")  # type: ignore
        if "y" not in by:  # type: ignore
            by.insert(0, "y")  # type: ignore

        df = df.drop(columns=["x", "y"]).rename(columns=dict(xBin="x", yBin="y"))
        res = _groupby_df(df, by, metrics, n_min)
        ds = res.to_xarray().squeeze()

        # change categorial index to coordinates
        for dim in ("x", "y"):
            ds[dim] = ds[dim].astype(float)
        return SkillGrid(ds)

    def scatter(
        self,
        *,
        bins=120,
        quantiles=None,
        fit_to_quantiles=False,
        show_points=None,
        show_hist=None,
        show_density=None,
        backend="matplotlib",
        figsize=(8, 8),
        xlim=None,
        ylim=None,
        reg_method="ols",
        title=None,
        xlabel=None,
        ylabel=None,
        skill_table=None,
        **kwargs,
    ):
        warnings.warn("scatter is deprecated, use plot.scatter instead", FutureWarning)

        # TODO remove in v1.1
        model, start, end, area = _get_deprecated_args(kwargs)
        observation, variable = _get_deprecated_obs_var_args(kwargs)

        # select model
        mod_id = _get_idx(model, self.mod_names)
        mod_name = self.mod_names[mod_id]

        # select variable
        var_id = _get_idx(variable, self.var_names)
        var_name = self.var_names[var_id]

        # filter data
        cmp = self.sel(
            model=mod_name,
            observation=observation,
            variable=var_name,
            start=start,
            end=end,
            area=area,
        )

        return cmp.plot.scatter(
            bins=bins,
            quantiles=quantiles,
            fit_to_quantiles=fit_to_quantiles,
            show_points=show_points,
            show_hist=show_hist,
            show_density=show_density,
            backend=backend,
            figsize=figsize,
            xlim=xlim,
            ylim=ylim,
            reg_method=reg_method,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            skill_table=skill_table,
            **kwargs,
        )

    def mean_skill(
        self,
        *,
        weights: Optional[Union[str, List[float], Dict[str, float]]] = None,
        metrics: Optional[list] = None,
        **kwargs,
    ) -> SkillTable:
        """Weighted mean of skills

        First, the skill is calculated per observation,
        the weighted mean of the skills is then found.

        .. warning::
            This method is NOT the mean skill of all observational points! (mean_skill_points)

        Parameters
        ----------
        weights : (str, List(float), Dict(str, float)), optional
            None: use observations weight attribute
            "equal": giving all observations equal weight,
            "points": giving all points equal weight,
            list of weights e.g. [0.3, 0.3, 0.4] per observation,
            dictionary of observations with special weigths, others will be set to 1.0
            by default None (i.e. observations weight attribute if assigned else "equal")
        metrics : list, optional
            list of modelskill.metrics, by default modelskill.options.metrics.list

        Returns
        -------
        SkillTable
            mean skill assessment as a skill object

        See also
        --------
        skill
            skill assessment per observation
        mean_skill_points
            skill assessment pooling all observation points together

        Examples
        --------
        >>> import modelskill as ms
        >>> cc = ms.match([HKNA,EPL,c2], mod=HKZN_local)
        >>> cc.mean_skill().round(2)
                      n  bias  rmse  urmse   mae    cc    si    r2
        HKZN_local  564 -0.09  0.31   0.28  0.24  0.97  0.09  0.99
        >>> s = cc.mean_skill(weights="equal")
        >>> s = cc.mean_skill(weights="points")
        >>> s = cc.mean_skill(weights={"EPL": 2.0}) # more weight on EPL, others=1.0
        """

        # TODO remove in v1.1
        model, start, end, area = _get_deprecated_args(kwargs)
        observation, variable = _get_deprecated_obs_var_args(kwargs)
        assert kwargs == {}, f"Unknown keyword arguments: {kwargs}"

        # filter data
        cmp = self.sel(
            model=model,  # deprecated
            observation=observation,  # deprecated
            variable=variable,  # deprecated
            start=start,  # deprecated
            end=end,  # deprecated
            area=area,  # deprecated
        )
        if cmp.n_points == 0:
            raise ValueError("Dataset is empty, no data to compare.")

        ## ---- end of deprecated code ----

        df = cmp.to_dataframe()
        mod_names = cmp.mod_names  # df.model.unique()
        # obs_names = cmp.obs_names  # df.observation.unique()
        var_names = cmp.var_names  # self.var_names

        # skill assessment
        metrics = _parse_metric(metrics, self.metrics, return_list=True)
        # s = self.skill(df=df, metrics=metrics)
        s = cmp.skill(metrics=metrics)
        if s is None:
            return None
        skilldf = s.to_dataframe()

        # weights
        weights = cmp._parse_weights(weights, s.obs_names)
        skilldf["weights"] = (
            skilldf.n if weights is None else np.tile(weights, len(mod_names))  # type: ignore
        )

        def weighted_mean(x):
            return np.average(x, weights=skilldf.loc[x.index, "weights"])

        # group by
        by = cmp._mean_skill_by(skilldf, mod_names, var_names)
        agg = {"n": "sum"}
        for metric in metrics:  # type: ignore
            agg[metric.__name__] = weighted_mean  # type: ignore
        res = skilldf.groupby(by).agg(agg)

        # TODO is this correct?
        res.index.name = "model"

        # output
        res = cmp._add_as_col_if_not_in_index(df, res, fields=["model", "variable"])
        return SkillTable(res.astype({"n": int}))

    # def mean_skill_points(
    #     self,
    #     *,
    #     metrics: Optional[list] = None,
    #     **kwargs,
    # ) -> Optional[SkillTable]:  # TODO raise error if no data?
    #     """Mean skill of all observational points

    #     All data points are pooled (disregarding which observation they belong to),
    #     the skill is then found (for each model).

    #     .. note::
    #         No weighting can be applied with this method,
    #         use mean_skill() if you need to apply weighting

    #     .. warning::
    #         This method is NOT the mean of skills (mean_skill)

    #     Parameters
    #     ----------
    #     metrics : list, optional
    #         list of modelskill.metrics, by default modelskill.options.metrics.list

    #     Returns
    #     -------
    #     SkillTable
    #         mean skill assessment as a skill object

    #     See also
    #     --------
    #     skill
    #         skill assessment per observation
    #     mean_skill
    #         weighted mean of skills (not the same as this method)

    #     Examples
    #     --------
    #     >>> import modelskill as ms
    #     >>> cc = ms.match(obs, mod)
    #     >>> cc.mean_skill_points()
    #     """

    #     # TODO remove in v1.1
    #     model, start, end, area = _get_deprecated_args(kwargs)
    #     observation, variable = _get_deprecated_obs_var_args(kwargs)
    #     assert kwargs == {}, f"Unknown keyword arguments: {kwargs}"

    #     # filter data
    #     cmp = self.sel(
    #         model=model,
    #         observation=observation,
    #         variable=variable,
    #         start=start,
    #         end=end,
    #         area=area,
    #     )
    #     if cmp.n_points == 0:
    #         warnings.warn("No data!")
    #         return None

    #     dfall = cmp.to_dataframe()
    #     dfall["observation"] = "all"

    #     # TODO: no longer possible to do this way
    #     # return self.skill(df=dfall, metrics=metrics)
    #     return cmp.skill(metrics=metrics)  # NOT CORRECT - SEE ABOVE

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
            observations = self.obs_names
        else:
            observations = [observations] if np.isscalar(observations) else observations
            observations = [_get_name(o, self.obs_names) for o in observations]
        n_obs = len(observations)

        if weights is None:
            # get weights from observation objects
            # default is equal weight to all
            weights = [self.comparers[o].weight for o in observations]
        else:
            if isinstance(weights, int):
                weights = np.ones(n_obs)  # equal weight to all
            elif isinstance(weights, dict):
                w_dict = weights
                weights = [w_dict.get(name, 1.0) for name in (self.obs_names)]

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
        if weights is not None:
            assert len(weights) == n_obs
        return weights

    def score(
        self,
        metric: str | Callable = mtr.rmse,
        **kwargs,
    ) -> Dict[str, float]:
        """Weighted mean score of model(s) over all observations

        Wrapping mean_skill() with a single metric.

        NOTE: will take simple mean over different variables

        Parameters
        ----------
        weights : (str, List(float), Dict(str, float)), optional
            None: use observations weight attribute
            "equal": giving all observations equal weight,
            "points": giving all points equal weight,
            list of weights e.g. [0.3, 0.3, 0.4] per observation,
            dictionary of observations with special weigths, others will be set to 1.0
            by default None (i.e. observations weight attribute if assigned else "equal")
        metric : list, optional
            a single metric from modelskill.metrics, by default rmse

        Returns
        -------
        Dict[str, float]
            mean of skills score as a single number (for each model)

        See also
        --------
        skill
            skill assessment per observation
        mean_skill
            weighted mean of skills assessment
        mean_skill_points
            skill assessment pooling all observation points together

        Examples
        --------
        >>> import modelskill as ms
        >>> cc = ms.match(obs, mod)
        >>> cc.score()
        0.30681206
        >>> cc.score(weights=[0.1,0.1,0.8])
        0.3383011631797379

        >>> cc.score(weights='points', metric="mape")
        8.414442957854142
        """

        weights = kwargs.pop("weights", None)
        metric = _parse_metric(metric, self.metrics)
        if not (callable(metric) or isinstance(metric, str)):
            raise ValueError("metric must be a string or a function")

        model, start, end, area = _get_deprecated_args(kwargs)
        observation, variable = _get_deprecated_obs_var_args(kwargs)
        assert kwargs == {}, f"Unknown keyword arguments: {kwargs}"

        if model is None:
            models = self.mod_names
        else:
            # TODO: these two lines looks familiar, extract to function
            models = [model] if np.isscalar(model) else model  # type: ignore
            models = [_get_name(m, self.mod_names) for m in models]  # type: ignore

        cmp = self.sel(
            model=models,  # deprecated
            observation=observation,  # deprecated
            variable=variable,  # deprecated
            start=start,  # deprecated
            end=end,  # deprecated
            area=area,  # deprecated
        )

        if cmp.n_points == 0:
            raise ValueError("Dataset is empty, no data to compare.")

        ## ---- end of deprecated code ----

        skill = cmp.mean_skill(weights=weights, metrics=[metric])
        df = skill.to_dataframe()

        metric_name = metric if isinstance(metric, str) else metric.__name__

        score = df[metric_name].to_dict()

        return score

    def taylor(
        self,
        normalize_std=False,
        aggregate_observations=True,
        figsize=(7, 7),
        marker="o",
        marker_size=6.0,
        title="Taylor diagram",
        **kwargs,
    ):
        warnings.warn("taylor is deprecated, use plot.taylor instead", FutureWarning)

        model, start, end, area = _get_deprecated_args(kwargs)
        observation, variable = _get_deprecated_obs_var_args(kwargs)
        assert kwargs == {}, f"Unknown keyword arguments: {kwargs}"

        cmp = self.sel(
            model=model,
            observation=observation,
            variable=variable,
            start=start,
            end=end,
            area=area,
        )

        if cmp.n_points == 0:
            warnings.warn("No data!")
            return

        if (not aggregate_observations) and (not normalize_std):
            raise ValueError(
                "aggregate_observations=False is only possible if normalize_std=True!"
            )

        metrics = [mtr._std_obs, mtr._std_mod, mtr.cc]
        skill_func = cmp.mean_skill if aggregate_observations else cmp.skill
        s = skill_func(metrics=metrics)

        df = s.to_dataframe()
        ref_std = 1.0 if normalize_std else df.iloc[0]["_std_obs"]

        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.map("_".join)

        df = df[["_std_obs", "_std_mod", "cc"]].copy()
        df.columns = ["obs_std", "std", "cc"]
        pts = [
            TaylorPoint(
                r.Index, r.obs_std, r.std, r.cc, marker=marker, marker_size=marker_size
            )
            for r in df.itertuples()
        ]

        taylor_diagram(
            obs_std=ref_std,
            points=pts,
            figsize=figsize,
            normalize_std=normalize_std,
            title=title,
        )

    def save(self, filename: Union[str, Path]) -> None:
        """Save the ComparerCollection to a zip file.

        Parameters
        ----------
        filename : str or Path
            Filename of the zip file.

        Examples
        --------
        >>> cc = ms.match(obs, mod)
        >>> cc.save("my_comparer_collection.msk")

        Notes
        -----
        Each comparer is stored as a netcdf file in the zip file.
        """

        files = []
        for name, cmp in self.comparers.items():
            cmp_fn = f"{name}.nc"
            cmp.save(cmp_fn)
            files.append(cmp_fn)

        with zipfile.ZipFile(filename, "w") as zip:
            for f in files:
                zip.write(f)
                os.remove(f)

    @staticmethod
    def load(filename: Union[str, Path]) -> "ComparerCollection":
        """Load a ComparerCollection from a zip file.

        Parameters
        ----------
        filename : str or Path
            Filename of the zip file.

        Returns
        -------
        ComparerCollection
            The loaded ComparerCollection.

        Examples
        --------
        >>> cc = ms.match(obs, mod)
        >>> cc.save("my_comparer_collection.msk")
        >>> cc2 = ms.ComparerCollection.load("my_comparer_collection.msk")
        """

        folder = tempfile.TemporaryDirectory().name

        with zipfile.ZipFile(filename, "r") as zip:
            for f in zip.namelist():
                if f.endswith(".nc"):
                    zip.extract(f, path=folder)

        comparers = [
            ComparerCollection._load_comparer(folder, f) for f in os.listdir(folder)
        ]
        return ComparerCollection(comparers)

    @staticmethod
    def _load_comparer(folder, f) -> Comparer:
        f = os.path.join(folder, f)
        cmp = Comparer.load(f)
        os.remove(f)
        return cmp

    def kde(self, ax=None, **kwargs):
        warnings.warn("kde is deprecated, use plot.kde instead", FutureWarning)

        return self.plot.kde(ax=ax, **kwargs)

    def hist(
        self,
        model=None,
        bins=100,
        title=None,
        density=True,
        alpha=0.5,
        **kwargs,
    ):
        warnings.warn("hist is deprecated, use plot.hist instead", FutureWarning)

        return self.plot.hist(
            model=model, bins=bins, title=title, density=density, alpha=alpha, **kwargs
        )
