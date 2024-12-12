from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import os
from pathlib import Path
import tempfile
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Union,
    Optional,
    Mapping,
    Iterable,
    overload,
    Hashable,
    Tuple,
)
import warnings
import zipfile
import numpy as np
import pandas as pd
import polars as pl



from ._collection_plotter import ComparerCollectionPlotter
from ..skill import SkillTable
from ..skill_grid import SkillGrid

from ..utils import _get_name
from ._comparison import Comparer, Scoreable
from ..metrics import _parse_metric
from ._utils import (
    _add_spatial_grid_to_df,
    _groupby_df,
    _parse_groupby,
    IdxOrNameTypes,
    TimeTypes,
)


class ComparerCollection(Mapping, Scoreable):
    """
    Collection of comparers, constructed by calling the `modelskill.match`
    method or by initializing with a list of comparers.

    NOTE: In case of multiple model results with different time coverage,
    only the _overlapping_ time period will be used! (intersection)

    Examples
    --------
    >>> import modelskill as ms
    >>> mr = ms.DfsuModelResult("Oresund2D.dfsu", item=0)
    >>> o1 = ms.PointObservation("klagshamn.dfs0", item=0, x=366844, y=6154291, name="Klagshamn")
    >>> o2 = ms.PointObservation("drogden.dfs0", item=0, x=355568.0, y=6156863.0)
    >>> cmp1 = ms.match(o1, mr)  # Comparer
    >>> cmp2 = ms.match(o2, mr)  # Comparer
    >>> ccA = ms.ComparerCollection([cmp1, cmp2])
    >>> ccB = ms.match(obs=[o1, o2], mod=mr)
    >>> sk = ccB.skill()
    >>> ccB["Klagshamn"].plot.timeseries()
    """

    plotter = ComparerCollectionPlotter

    def __init__(self, comparers: Iterable[Comparer]) -> None:
        self._comparers: Dict[str, Comparer] = {}

        for cmp in comparers:
            if cmp.name in self._comparers:
                # comparer with this name already exists!
                # maybe the user is trying to add a new model
                # or a new time period
                self._comparers[cmp.name] += cmp
            else:
                self._comparers[cmp.name] = cmp

        self.plot = ComparerCollection.plotter(self)
        """Plot using the ComparerCollectionPlotter

        Examples
        --------
        >>> cc.plot.scatter()
        >>> cc.plot.kde()
        >>> cc.plot.taylor()
        >>> cc.plot.hist()
        """

    @property
    def _name(self) -> str:
        return "Observations"

    @property
    def _unit_text(self) -> str:
        # Picking the first one is arbitrary, but it should be the same for all
        # we could check that they are all the same, but let's assume that they are
        # for cmp in self:
        #     if cmp._unit_text != text:
        #         warnings.warn(f"Unit text is inconsistent: {text} vs {cmp._unit_text}")
        return self[0]._unit_text

    @property
    def n_points(self) -> int:
        """number of compared points"""
        return sum([c.n_points for c in self._comparers.values()])

    @property
    def start_time(self) -> pd.Timestamp:
        """start timestamp of compared data"""
        starts = [pd.Timestamp.max]
        for cmp in self._comparers.values():
            starts.append(cmp.time[0])
        return min(starts)

    @property
    def end_time(self) -> pd.Timestamp:
        """end timestamp of compared data"""
        ends = [pd.Timestamp.min]
        for cmp in self._comparers.values():
            ends.append(cmp.time[-1])
        return max(ends)

    @property
    def obs_names(self) -> List[str]:
        """List of observation names"""
        return [c.name for c in self._comparers.values()]

    @property
    def n_observations(self) -> int:
        """Number of observations (same as len(cc))"""
        return len(self)

    @property
    def mod_names(self) -> List[str]:
        """List of unique model names"""
        all_names = [n for cmp in self for n in cmp.mod_names]
        # preserve order (instead of using set)
        return list(dict.fromkeys(all_names))

    @property
    def n_models(self) -> int:
        """Number of unique models"""
        return len(self.mod_names)

    @property
    def aux_names(self) -> List[str]:
        """List of unique auxiliary names"""
        all_names = [n for cmp in self for n in cmp.aux_names]
        # preserve order (instead of using set)
        return list(dict.fromkeys(all_names))

    @property
    def quantity_names(self) -> List[str]:
        """List of unique quantity names"""
        all_names = [cmp.quantity.name for cmp in self]
        # preserve order (instead of using set)
        return list(dict.fromkeys(all_names))

    @property
    def n_quantities(self) -> int:
        """Number of unique quantities"""
        return len(self.quantity_names)

    def __repr__(self) -> str:
        out = []
        out.append("<ComparerCollection>")
        out.append("Comparers:")
        for index, (key, value) in enumerate(self._comparers.items()):
            out.append(f"{index}: {key} - {value.quantity}")
        return str.join("\n", out)

    def rename(self, mapping: Dict[str, str]) -> "ComparerCollection":
        """Rename observation, model or auxiliary data variables

        Parameters
        ----------
        mapping : dict
            mapping of old names to new names

        Returns
        -------
        ComparerCollection

        Examples
        --------
        >>> cc = ms.match([o1, o2], [mr1, mr2])
        >>> cc.mod_names
        ['mr1', 'mr2']
        >>> cc2 = cc.rename({'mr1': 'model1'})
        >>> cc2.mod_names
        ['model1', 'mr2']
        """
        for k in mapping.keys():
            allowed_keys = self.obs_names + self.mod_names + self.aux_names
            if k not in allowed_keys:
                raise KeyError(f"Unknown key: {k}; must be one of {allowed_keys}")

        cmps = []
        for cmp in self._comparers.values():
            cmps.append(cmp.rename(mapping, errors="ignore"))
        return ComparerCollection(cmps)

    @overload
    def __getitem__(self, x: slice | Iterable[Hashable]) -> ComparerCollection: ...

    @overload
    def __getitem__(self, x: int | Hashable) -> Comparer: ...

    def __getitem__(
        self, x: int | Hashable | slice | Iterable[Hashable]
    ) -> Comparer | ComparerCollection:
        if isinstance(x, str):
            return self._comparers[x]

        if isinstance(x, slice):
            idxs = list(range(*x.indices(len(self))))
            return ComparerCollection([self[i] for i in idxs])

        if isinstance(x, int):
            name = _get_name(x, self.obs_names)
            return self._comparers[name]

        if isinstance(x, Iterable):
            cmps = [self[i] for i in x]
            return ComparerCollection(cmps)

        raise TypeError(f"Invalid type for __getitem__: {type(x)}")

    def __len__(self) -> int:
        return len(self._comparers)

    def __iter__(self) -> Iterator[Comparer]:
        return iter(self._comparers.values())

    def copy(self) -> "ComparerCollection":
        return deepcopy(self)

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
        model: Optional[IdxOrNameTypes] = None,
        observation: Optional[IdxOrNameTypes] = None,
        quantity: Optional[IdxOrNameTypes] = None,
        start: Optional[TimeTypes] = None,
        end: Optional[TimeTypes] = None,
        time: Optional[TimeTypes] = None,
        area: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> "ComparerCollection":
        """Select data based on model, time and/or area.

        Parameters
        ----------
        model : str or int or list of str or list of int, optional
            Model name or index. If None, all models are selected.
        observation : str or int or list of str or list of int, optional
            Observation name or index. If None, all observations are selected.
        quantity : str or int or list of str or list of int, optional
            Quantity name or index. If None, all quantities are selected.
        start : str or datetime, optional
            Start time. If None, all times are selected.
        end : str or datetime, optional
            End time. If None, all times are selected.
        time : str or datetime, optional
            Time. If None, all times are selected.
        area : list of float, optional
            bbox: [x0, y0, x1, y1] or Polygon. If None, all areas are selected.
        **kwargs
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

        if (quantity is not None) and (self.n_quantities > 1):
            quantity = [quantity] if np.isscalar(quantity) else quantity  # type: ignore
            quantity = [_get_name(v, self.quantity_names) for v in quantity]  # type: ignore
        else:
            quantity = self.quantity_names

        cmps = []
        for cmp in self._comparers.values():
            if cmp.name in observation and cmp.quantity.name in quantity:
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

    def filter_by_attrs(self, **kwargs: Any) -> "ComparerCollection":
        """Filter by comparer attrs similar to xarray.Dataset.filter_by_attrs

        Parameters
        ----------
        **kwargs
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
        for cmp in self._comparers.values():
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
        q_cmps = [cmp.query(query) for cmp in self._comparers.values()]
        cmps_with_data = [cmp for cmp in q_cmps if cmp.n_points > 0]

        return ComparerCollection(cmps_with_data)

    def skill(
        self,
        by: str | Iterable[str] | None = None,
        metrics: Iterable[str] | Iterable[Callable] | str | Callable | None = None,
        observed: bool = False,
    ) -> SkillTable:
        """Aggregated skill assessment of model(s)

        Parameters
        ----------
        by : str or List[str], optional
            group by, by default ["model", "observation"]

            - by column name
            - by temporal bin of the DateTimeIndex via the freq-argument
            (using pandas pd.Grouper(freq)), e.g.: 'freq:M' = monthly; 'freq:D' daily
            - by the dt accessor of the DateTimeIndex (e.g. 'dt.month') using the
            syntax 'dt:month'. The dt-argument is different from the freq-argument
            in that it gives month-of-year rather than month-of-data.
            - by attributes, stored in the cc.data.attrs container,
            e.g.: 'attrs:obs_provider' = group by observation provider or
            'attrs:gtype' = group by geometry type (track or point)
        metrics : list, optional
            list of modelskill.metrics (or str), by default modelskill.options.metrics.list
        observed: bool, optional
            This only applies if any of the groupers are Categoricals.

            - True: only show observed values for categorical groupers.
            - False: show all values for categorical groupers.

        Returns
        -------
        SkillTable
            skill assessment as a SkillTable object

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

        >>> cc.sel(observation='c2', start='2017-10-28').skill().round(2)
                       n  bias  rmse  urmse   mae    cc    si    r2
        observation
        c2            41  0.33  0.41   0.25  0.36  0.96  0.06  0.99

        >>> cc.skill(by='freq:D').round(2)
                      n  bias  rmse  urmse   mae    cc    si    r2
        2017-10-27  239 -0.15  0.25   0.21  0.20  0.72  0.10  0.98
        2017-10-28  162 -0.07  0.19   0.18  0.16  0.96  0.06  1.00
        2017-10-29  163 -0.21  0.52   0.47  0.42  0.79  0.11  0.99
        """
        cc = self

        pmetrics = _parse_metric(metrics)
        # TODO don't use hardcoded metrics
        # pmetrics = ["n", "bias", "rmse", "urmse", "mae"]

        agg_cols = _parse_groupby(by, n_mod=cc.n_models, n_qnt=cc.n_quantities)
        agg_cols, attrs_keys = self._attrs_keys_in_by(agg_cols)

        df = cc._to_long_dataframe(attrs_keys=attrs_keys, observed=observed)
        assert "model" in df.columns

        res = _groupby_df(df, by=agg_cols, metrics=pmetrics)
        res = self._append_xy_to_res(res, cc)

        # TODO this should not be necessary
        if "model" not in res.columns:
            res = res.with_columns(pl.lit(cc.mod_names[0]).alias("model"))
        return SkillTable(res)

    def _to_long_dataframe(
        self, attrs_keys: Iterable[str] | None = None, observed: bool = False
    ) -> pl.DataFrame:
        """Return a copy of the data as a long-format pandas DataFrame (for groupby operations)"""
        frames = []
        for cmp in self:
            frame = cmp._to_long_dataframe(attrs_keys=attrs_keys)

            if self.n_quantities > 1:
                frame = frame.with_columns(pl.lit(cmp.quantity.name).alias("quantity"))
            frames.append(frame)

        # convert all ints and floats to f64
        import polars.selectors as cs

        # frames = [
        #    df.with_columns([pl.col(pl.Float32, pl.Int32, pl.Int64).cast(pl.Float64)])
        #    for df in frames
        # ]
        frames = [df.with_columns(cs.numeric().cast(pl.Float64)) for df in frames]

        # TODO why doesn't all frames have the same columns?
        res = pl.concat(frames, how="diagonal")
        if observed and attrs_keys is not None:
            # res = res.loc[~(res == False).any(axis=1)]  # noqa
            res = res.filter(pl.col(attrs_keys).is_not_null())

        return res

    @staticmethod
    def _attrs_keys_in_by(by: List[str | pd.Grouper]) -> Tuple[List[str], List[str]]:
        attrs_keys: List[str] = []
        agg_cols: List[str] = []
        for b in by:
            if isinstance(b, str) and b.startswith("attrs:"):
                key = b.split(":")[1]
                attrs_keys.append(key)
                agg_cols.append(key)
            else:
                agg_cols.append(b)
        return agg_cols, attrs_keys

    @staticmethod
    def _append_xy_to_res(res: pl.DataFrame, cc: ComparerCollection) -> pl.DataFrame:
        """skill() helper: Append x and y to res if possible"""

        if "observation" not in res.columns:
            return res
        xs = defaultdict(lambda: np.nan)
        ys = defaultdict(lambda: np.nan)
        for cmp in cc:
            if cmp.gtype == "point":
                xs[cmp.name] = float(cmp.x)
                ys[cmp.name] = float(cmp.y)

        # add x and y to res based on observation name based on the xs and ys dicts
        res = res.with_columns(
            pl.col("observation").map_elements(lambda name: xs.get(name)).alias("x"),
            pl.col("observation").map_elements(lambda name: ys.get(name)).alias("y"),
        )

        return res

    def _add_as_col_if_not_in_index(
        self,
        df: pd.DataFrame,
        skilldf: pd.DataFrame,
        fields: List[str] = ["model", "observation", "quantity"],
    ) -> pd.DataFrame:
        """skill() helper: Add a field to skilldf if unique in df"""
        for field in reversed(fields):
            if (field == "model") and (self.n_models <= 1):
                continue
            if (field == "quantity") and (self.n_quantities <= 1):
                continue
            if field not in skilldf.index.names:
                unames = df[field].unique()
                if len(unames) == 1:
                    skilldf.insert(loc=0, column=field, value=unames[0])
        return skilldf

    def gridded_skill(
        self,
        bins: int = 5,
        binsize: float | None = None,
        by: str | Iterable[str] | None = None,
        metrics: Iterable[str] | Iterable[Callable] | str | Callable | None = None,
        n_min: Optional[int] = None,
        **kwargs: Any,
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
        by : str, List[str], optional
            group by, by default ["model", "observation"]

            - by column name
            - by temporal bin of the DateTimeIndex via the freq-argument
            (using pandas pd.Grouper(freq)), e.g.: 'freq:M' = monthly; 'freq:D' daily
            - by the dt accessor of the DateTimeIndex (e.g. 'dt.month') using the
            syntax 'dt:month'. The dt-argument is different from the freq-argument
            in that it gives month-of-year rather than month-of-data.
        metrics : list, optional
            list of modelskill.metrics, by default modelskill.options.metrics.list
        n_min : int, optional
            minimum number of observations in a grid cell;
            cells with fewer observations get a score of `np.nan`

        Returns
        -------
        SkillGrid
            skill assessment as a SkillGrid object

        See also
        --------
        skill
            a method for aggregated skill assessment

        Examples
        --------
        >>> import modelskill as ms
        >>> cc = ms.match([HKNA,EPL,c2], mr)  # with satellite track measurements
        >>> gs = cc.gridded_skill(metrics='bias')
        >>> gs.data
        <xarray.Dataset>
        Dimensions:      (x: 5, y: 5)
        Coordinates:
            observation   'alti'
        * x            (x) float64 -0.436 1.543 3.517 5.492 7.466
        * y            (y) float64 50.6 51.66 52.7 53.75 54.8
        Data variables:
            n            (x, y) int32 3 0 0 14 37 17 50 36 72 ... 0 0 15 20 0 0 0 28 76
            bias         (x, y) float64 -0.02626 nan nan ... nan 0.06785 -0.1143

        >>> gs = cc.gridded_skill(binsize=0.5)
        >>> gs.data.coords
        Coordinates:
            observation   'alti'
        * x            (x) float64 -1.5 -0.5 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5
        * y            (y) float64 51.5 52.5 53.5 54.5 55.5 56.5
        """
        cmp = self

        metrics = _parse_metric(metrics)

        # TODO avoid hardcoded metrics
        # metrics = ["n", "bias", "rmse", "mae"]

        df = cmp._to_long_dataframe()
        df = _add_spatial_grid_to_df(df=df, bins=bins, binsize=binsize)

        agg_cols = _parse_groupby(by, n_mod=cmp.n_models, n_qnt=cmp.n_quantities)
        if "x" not in agg_cols:
            agg_cols.insert(0, "x")
        if "y" not in agg_cols:
            agg_cols.insert(0, "y")

        df = df.drop(["x", "y"]).rename(dict(xBin="x", yBin="y"))
        res = _groupby_df(df, by=agg_cols, metrics=metrics, n_min=n_min)  # type: ignore

        potential_cols = ["x", "y", "model", "observation"]
        cols = [c for c in potential_cols if c in res.columns]

        ds = res.to_pandas().set_index(cols).to_xarray().squeeze()

        # change categorial index to coordinates
        # for dim in ("x", "y"):
        #    ds[dim] = ds[dim].astype(float)
        return SkillGrid(ds)

    def mean_skill(
        self,
        *,
        weights: Optional[Union[str, List[float], Dict[str, float]]] = None,
        metrics: Optional[list] = None,
        **kwargs: Any,
    ) -> SkillTable:
        """Weighted mean of skills

        First, the skill is calculated per observation,
        the weighted mean of the skills is then found.

        Warning: This method is NOT the mean skill of
        all observational points! (mean_skill_points)

        Parameters
        ----------
        weights : str or List(float) or Dict(str, float), optional
            weighting of observations, by default None

            - None: use observations weight attribute (if assigned, else "equal")
            - "equal": giving all observations equal weight,
            - "points": giving all points equal weight,
            - list of weights e.g. [0.3, 0.3, 0.4] per observation,
            - dictionary of observations with special weigths, others will be set to 1.0
        metrics : list, optional
            list of modelskill.metrics, by default modelskill.options.metrics.list

        Returns
        -------
        SkillTable
            mean skill assessment as a SkillTable object

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
        >>> sk = cc.mean_skill(weights="equal")
        >>> sk = cc.mean_skill(weights="points")
        >>> sk = cc.mean_skill(weights={"EPL": 2.0}) # more weight on EPL, others=1.0
        """

        cc = self

        df = cc._to_long_dataframe()  # TODO: remove
        assert "model" in df.columns
        mod_names = cc.mod_names
        # obs_names = cmp.obs_names  # df.observation.unique()
        qnt_names = cc.quantity_names

        # skill assessment
        pmetrics = _parse_metric(metrics)

        # TODO avoid hardcoded metrics
        # pmetrics = ["n", "bias", "rmse", "mae"]
        sk = cc.skill(metrics=pmetrics)
        if sk is None:
            return None
        skilldf = sk.to_dataframe()
        assert "model" in skilldf.columns

        # weights
        weights = cc._parse_weights(weights, sk.obs_names)
        # skilldf["weights"] = (
        #    skilldf.n if weights is None else np.tile(weights, len(mod_names))  # type: ignore
        # )
        if weights is None:
            skilldf = skilldf.with_columns(pl.col("n").alias("weights"))
        else:
            wdict = {o: w for o, w in zip(sk.obs_names, weights)}
            skilldf = skilldf.with_columns(
                pl.col("observation")
                .map_elements(lambda name: wdict.get(name))
                .alias("weights")
            )

        # def weighted_mean(x: Any) -> Any:
        #    return np.average(x, weights=skilldf.loc[x.index, "weights"])
        # weighted_mean = pl.

        # group by
        by = cc._mean_skill_by(skilldf, mod_names, qnt_names)  # type: ignore
        # agg = {"n": "sum"}
        # for metric in pmetrics:  # type: ignore
        #    agg[metric.__name__] = weighted_mean  # type: ignore
        # (pl.col("values") * pl.col("weights")).sum() / pl.col("weights").sum()).alias("weighted_mean")
        exprs = [
            (pl.col(metric) * pl.col("weights")).sum() / pl.col("weights").sum()
            for metric in pmetrics
        ]
        # res = skilldf.groupby(by, observed=False).agg(agg)

        # Numpy ufuncs are supported, e.g.
        # df.select(np.log(pl.all()).name.suffix("_log"))
        res = skilldf.group_by(by).agg(exprs)

        # res = skilldf.group_by(by).agg(np.average(pl.all(), weights=pl.col("weights")))

        # TODO is this correct?
        # res.index.name = "model"

        # output
        # res = cc._add_as_col_if_not_in_index(df, res, fields=["model", "quantity"])  # type: ignore
        return SkillTable(res)
        # return SkillTable(res.astype({"n": int}))

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

    #     cmp = self
    #     dfall = cmp.to_dataframe()
    #     dfall["observation"] = "all"

    #     # TODO: no longer possible to do this way
    #     # return self.skill(df=dfall, metrics=metrics)
    #     return cmp.skill(metrics=metrics)  # NOT CORRECT - SEE ABOVE

    def _mean_skill_by(self, skilldf, mod_names, qnt_names):  # type: ignore
        # TODO clean up this mess
        by = ["model"]
        if len(mod_names) > 1:
            by.append("model")
        if len(qnt_names) > 1:
            by.append("quantity")
        if len(by) == 0:
            if (self.n_quantities > 1) and ("quantity" in skilldf):
                by.append("quantity")
            elif "model" in skilldf:
                by.append("model")
            else:
                # by = [mod_names[0]] * len(skilldf)
                by = None
        return list(set(by))

    # TODO add useful type hints
    def _parse_weights(self, weights: Any, observations: Any) -> Any:
        if observations is None:
            observations = self.obs_names
        else:
            observations = [observations] if np.isscalar(observations) else observations
            observations = [_get_name(o, self.obs_names) for o in observations]
        n_obs = len(observations)

        if weights is None:
            # get weights from observation objects
            # default is equal weight to all
            weights = [self._comparers[o].weight for o in observations]
        else:
            if isinstance(weights, int):
                weights = np.ones(n_obs)  # equal weight to all
            elif isinstance(weights, dict):
                w_dict = weights
                weights = [w_dict.get(name, 1.0) for name in observations]

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
        metric: str | Callable = "rmse",
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Weighted mean score of model(s) over all observations

        Wrapping mean_skill() with a single metric.

        NOTE: will take simple mean over different quantities!

        Parameters
        ----------
        weights : str or List(float) or Dict(str, float), optional
            weighting of observations, by default None

            - None: use observations weight attribute (if assigned, else "equal")
            - "equal": giving all observations equal weight,
            - "points": giving all points equal weight,
            - list of weights e.g. [0.3, 0.3, 0.4] per observation,
            - dictionary of observations with special weigths, others will be set to 1.0
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
        >>> cc = ms.match([o1, o2], mod)
        >>> cc.score()
        {'mod': 0.30681206}
        >>> cc.score(weights=[0.1,0.1,0.8])
        {'mod': 0.3383011631797379}

        >>> cc.score(weights='points', metric="mape")
        {'mod': 8.414442957854142}
        """

        weights = kwargs.pop("weights", None)

        metric = _parse_metric(metric)[0]

        if weights is None:
            weights = {c.name: c.weight for c in self._comparers.values()}

        if not (callable(metric) or isinstance(metric, str)):
            raise ValueError("metric must be a string or a function")

        cmp = self

        if cmp.n_points == 0:
            raise ValueError("Dataset is empty, no data to compare.")

        sk = cmp.mean_skill(weights=weights, metrics=[metric])
        assert isinstance(metric, str)

        return {k: v for k, v in sk.sort("model").select(["model", metric]).rows()}

    def save(self, filename: Union[str, Path]) -> None:
        """Save the ComparerCollection to a zip file.

        Each comparer is stored as a netcdf file in the zip file.

        Parameters
        ----------
        filename : str or Path
            Filename of the zip file.

        Examples
        --------
        >>> cc = ms.match(obs, mod)
        >>> cc.save("my_comparer_collection.msk")
        """

        files = []
        no = 0
        for name, cmp in self._comparers.items():
            cmp_fn = f"{no}_{name}.nc"
            cmp.save(cmp_fn)
            files.append(cmp_fn)
            no += 1

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
            ComparerCollection._load_comparer(folder, f)
            for f in sorted(os.listdir(folder))
        ]
        return ComparerCollection(comparers)

    @staticmethod
    def _load_comparer(folder: str, f: str) -> Comparer:
        f = os.path.join(folder, f)
        cmp = Comparer.load(f)
        os.remove(f)
        return cmp

    