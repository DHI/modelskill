from pathlib import Path
from typing import Dict, List, Optional, Union, Iterable, Sequence
import warnings
from inspect import getmembers, isfunction
from matplotlib.axes import Axes  # type: ignore
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from copy import deepcopy

from .. import metrics as mtr
from .. import Quantity
from .. import __version__
from ..observation import Observation, PointObservation, TrackObservation

from ._comparer_plotter import ComparerPlotter
from ..skill import AggregatedSkill
from ..spatial import SpatialSkill
from ..settings import options, register_option, reset_option

from ._utils import _get_name

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._collection import ComparerCollection


# TODO remove in v1.1
def _get_deprecated_args(kwargs):
    model, start, end, area = None, None, None, None

    # Don't bother refactoring this, it will be removed in v1.1
    if "model" in kwargs:
        model = kwargs.pop("model")
        if model is not None:
            warnings.warn(
                f"The 'model' argument is deprecated, use 'sel(model='{model}')' instead",
                FutureWarning,
            )

    if "start" in kwargs:
        start = kwargs.pop("start")

        if start is not None:
            warnings.warn(
                f"The 'start' argument is deprecated, use 'sel(start={start})' instead",
                FutureWarning,
            )

    if "end" in kwargs:
        end = kwargs.pop("end")

        if end is not None:
            warnings.warn(
                f"The 'end' argument is deprecated, use 'sel(end={end})' instead",
                FutureWarning,
            )

    if "area" in kwargs:
        area = kwargs.pop("area")

        if area is not None:
            warnings.warn(
                f"The 'area' argument is deprecated, use 'sel(area={area})' instead",
                FutureWarning,
            )

    return model, start, end, area


def _validate_metrics(metrics) -> None:
    for m in metrics:
        if isinstance(m, str):
            if m not in mtr.DEFINED_METRICS:
                raise ValueError(
                    f"Metric '{m}' is not defined. Valid metrics are {mtr.DEFINED_METRICS}"
                )


register_option(
    key="metrics.list",
    defval=[mtr.bias, mtr.rmse, mtr.urmse, mtr.mae, mtr.cc, mtr.si, mtr.r2],
    validator=_validate_metrics,
    doc="Default metrics list to be used in skill tables if specific metrics are not provided.",
)


MOD_COLORS = (
    "#1f78b4",
    "#33a02c",
    "#ff7f00",
    "#93509E",
    "#63CEFF",
    "#fdbf6f",
    "#004165",
    "#8B8D8E",
    "#0098DB",
    "#61C250",
    "#a6cee3",
    "#b2df8a",
    "#fb9a99",
    "#cab2d6",
    "#003f5c",
    "#2f4b7c",
    "#665191",
    "#e31a1c",
)


TimeDeltaTypes = Union[float, int, np.timedelta64, pd.Timedelta, timedelta]
TimeTypes = Union[str, np.datetime64, pd.Timestamp, datetime]
IdOrNameTypes = Union[int, str, List[int], List[str]]


def _interp_time(df: pd.DataFrame, new_time: pd.DatetimeIndex) -> pd.DataFrame:
    """Interpolate time series to new time index"""
    new_df = (
        df.reindex(df.index.union(new_time))
        .interpolate(method="time", limit_area="inside")
        .reindex(new_time)
    )
    return new_df


def _time_delta_to_pd_timedelta(time_delta: TimeDeltaTypes) -> pd.Timedelta:
    if isinstance(time_delta, (timedelta, np.timedelta64)):
        time_delta = pd.Timedelta(time_delta)
    elif np.isscalar(time_delta):
        # assume seconds
        time_delta = pd.Timedelta(time_delta, "s")  # type: ignore
    assert isinstance(time_delta, pd.Timedelta)
    return time_delta


def _remove_model_gaps(
    df: pd.DataFrame,
    mod_index: pd.DatetimeIndex,
    max_gap: TimeDeltaTypes,
) -> pd.DataFrame:
    """Remove model gaps longer than max_gap from dataframe"""
    max_gap = _time_delta_to_pd_timedelta(max_gap)
    valid_time = _get_valid_query_time(mod_index, df.index, max_gap)
    return df.loc[valid_time]


def _get_valid_query_time(
    mod_index: pd.DatetimeIndex, obs_index: pd.DatetimeIndex, max_gap: pd.Timedelta
):
    """Used only by _remove_model_gaps"""
    # init dataframe of available timesteps and their index
    df = pd.DataFrame(index=mod_index)
    df["idx"] = range(len(df))

    # for query times get available left and right index of source times
    df = _interp_time(df, obs_index).dropna()
    df["idxa"] = np.floor(df.idx).astype(int)
    df["idxb"] = np.ceil(df.idx).astype(int)

    # time of left and right source times and time delta
    df["ta"] = mod_index[df.idxa]
    df["tb"] = mod_index[df.idxb]
    df["dt"] = df.tb - df.ta

    # valid query times where time delta is less than max_gap
    valid_idx = df.dt <= max_gap
    return valid_idx


def _parse_metric(metric, default_metrics, return_list=False):
    if metric is None:
        metric = default_metrics

    if isinstance(metric, str):
        valid_metrics = [x[0] for x in getmembers(mtr, isfunction) if x[0][0] != "_"]

        if metric.lower() in valid_metrics:
            metric = getattr(mtr, metric.lower())
        else:
            raise ValueError(
                f"Invalid metric: {metric}. Valid metrics are {valid_metrics}."
            )
    elif isinstance(metric, Iterable):
        metrics = [_parse_metric(m, default_metrics) for m in metric]
        return metrics
    elif not callable(metric):
        raise TypeError(f"Invalid metric: {metric}. Must be either string or callable.")
    if return_list:
        if callable(metric) or isinstance(metric, str):
            metric = [metric]
    return metric


def _area_is_bbox(area) -> bool:
    is_bbox = False
    if area is not None:
        if not np.isscalar(area):
            area = np.array(area)
            if (area.ndim == 1) & (len(area) == 4):
                if np.all(np.isreal(area)):
                    is_bbox = True
    return is_bbox


def _area_is_polygon(area) -> bool:
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


def _inside_polygon(polygon, xy) -> bool:
    import matplotlib.path as mp  # type: ignore

    if polygon.ndim == 1:
        polygon = np.column_stack((polygon[0::2], polygon[1::2]))
    return mp.Path(polygon).contains_points(xy)


def _add_spatial_grid_to_df(
    df: pd.DataFrame, bins, binsize: Optional[float]
) -> pd.DataFrame:
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


def _groupby_df(df, by, metrics, n_min: Optional[int] = None):
    def calc_metrics(x):
        row = {}
        row["n"] = len(x)
        for metric in metrics:
            row[metric.__name__] = metric(x.obs_val, x.mod_val)
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


def _parse_groupby(by, n_models, n_obs, n_var=1):
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
        by = [_parse_groupby(b, n_models, n_obs, n_var) for b in by]
        return by
    else:
        raise ValueError("Invalid by argument. Must be string or list of strings.")
    return by


class Comparer:
    data: xr.Dataset
    raw_mod_data: Dict[str, pd.DataFrame]
    _obs_name = "Observation"
    plotter = ComparerPlotter

    def __init__(
        self,
        observation=None,
        modeldata=None,
        max_model_gap: Optional[TimeDeltaTypes] = None,
        matched_data: Optional[xr.Dataset] = None,
        raw_mod_data: Optional[Dict[str, pd.DataFrame]] = None,
    ):

        self.plot = Comparer.plotter(self)

        # TODO extract method
        if matched_data is not None:
            assert "Observation" in matched_data.data_vars

            # no missing values allowed in Observation
            if matched_data["Observation"].isnull().any():
                raise ValueError("Observation data must not contain missing values.")

            for key in matched_data.data_vars:
                if "kind" not in matched_data[key].attrs:
                    matched_data[key].attrs["kind"] = "auxiliary"
            if "x" not in matched_data:
                matched_data["x"] = np.nan
                matched_data["x"].attrs["kind"] = "position"

            if "y" not in matched_data:
                matched_data["y"] = np.nan
                matched_data["y"].attrs["kind"] = "position"

            if "color" not in matched_data["Observation"].attrs:
                matched_data["Observation"].attrs["color"] = "black"

            if "variable_name" not in matched_data.attrs:
                matched_data.attrs["variable_name"] = Quantity.undefined().name

            if "unit" not in matched_data["Observation"].attrs:
                matched_data["Observation"].attrs["unit"] = Quantity.undefined().unit

            self.data = matched_data
            self.raw_mod_data = (
                raw_mod_data
                if raw_mod_data is not None
                else {
                    key: value.to_dataframe()
                    for key, value in matched_data.data_vars.items()
                    if value.attrs["kind"] == "model"
                }
            )
            # TODO get quantity from matched_data object
            self.quantity: Quantity = Quantity.undefined()
        else:
            self.raw_mod_data = (
                self._parse_modeldata_list(modeldata) if modeldata is not None else {}
            )

            self.data = self._initialise_comparer(observation, max_model_gap)
            self.quantity: Quantity = observation.quantity

    def _mask_model_outside_observation_track(self, name, df_mod, df_obs) -> None:
        if len(df_mod) == 0:
            return
        if len(df_mod) != len(df_obs):
            raise ValueError("model and observation data must have same length")

        mod_xy = df_mod[["x", "y"]]
        obs_xy = df_obs[["x", "y"]]
        d_xy = np.sqrt(np.sum((obs_xy - mod_xy) ** 2, axis=1))
        # TODO why not use a fixed tolerance?
        tol_xy = self._minimal_accepted_distance(obs_xy)
        mask = d_xy > tol_xy
        df_mod.loc[mask, name] = np.nan
        if all(mask):
            warnings.warn("no (spatial) overlap between model and observation points")

    def _initialise_comparer(self, observation, max_model_gap) -> xr.Dataset:
        assert isinstance(observation, (PointObservation, TrackObservation))
        gtype = "point" if isinstance(observation, PointObservation) else "track"
        observation = deepcopy(observation)
        observation.trim(self._mod_start, self._mod_end)

        first = True
        for name, mdata in self.raw_mod_data.items():
            df = self._model2obs_interp(observation, mdata, max_model_gap)
            if gtype == "track":
                # TODO why is it necessary to do mask here? Isn't it an error if the model data is outside the observation track?
                self._mask_model_outside_observation_track(name, df, observation.data)

            if first:
                data = df
            else:
                data[name] = df[name]

            first = False

        data.index.name = "time"
        data = data.dropna()
        data = data.to_xarray()
        data.attrs["gtype"] = gtype

        if gtype == "point":
            data["x"] = observation.x
            data["y"] = observation.y
            data["z"] = observation.z

        data.attrs["name"] = observation.name
        # data.attrs["variable_name"] = observation.variable_name
        data.attrs["variable_name"] = observation.quantity.name
        data["x"].attrs["kind"] = "position"
        data["y"].attrs["kind"] = "position"
        data[self._obs_name].attrs["kind"] = "observation"
        data[self._obs_name].attrs["unit"] = observation.quantity.unit
        data[self._obs_name].attrs["color"] = observation.color
        data[self._obs_name].attrs["weight"] = observation.weight
        for n in self.mod_names:
            data[n].attrs["kind"] = "model"

        data.attrs["modelskill_version"] = __version__

        return data

    @staticmethod
    def _minimal_accepted_distance(obs_xy):
        # all consequtive distances
        vec = np.sqrt(np.sum(np.diff(obs_xy, axis=0), axis=1) ** 2)
        # fraction of small quantile
        return 0.5 * np.quantile(vec, 0.1)

    def _parse_modeldata_list(self, modeldata) -> Dict[str, pd.DataFrame]:
        """Convert to dict of dataframes"""
        if not isinstance(modeldata, Sequence):
            modeldata = [modeldata]

        mod_dfs = [self._parse_single_modeldata(m) for m in modeldata]
        return {m.columns[-1]: m for m in mod_dfs if m is not None}

    @staticmethod
    def _parse_single_modeldata(modeldata) -> pd.DataFrame:
        """Convert to dataframe and set index to pd.DatetimeIndex"""
        if hasattr(modeldata, "to_dataframe"):
            mod_df = modeldata.to_dataframe()
        elif isinstance(modeldata, pd.DataFrame):
            mod_df = modeldata
        else:
            raise ValueError(
                f"Unknown modeldata type '{type(modeldata)}' (mikeio.Dataset, xr.DataArray, xr.Dataset or pd.DataFrame)"
            )

        if not isinstance(mod_df.index, pd.DatetimeIndex):
            raise ValueError(
                "Modeldata index must be datetime-like (pd.DatetimeIndex, pd.to_datetime)"
            )

        time = mod_df.index.round(freq="100us")  # 0.0001s accuracy
        mod_df.index = pd.DatetimeIndex(time, freq="infer")
        return mod_df

    @classmethod
    def from_compared_data(cls, data, raw_mod_data=None):
        """Initialize from compared data"""
        return cls(matched_data=data, raw_mod_data=raw_mod_data)

    def __repr__(self):
        out = [
            f"<{type(self).__name__}>",
            f"Quantity: {self.quantity}",
            f"Observation: {self.name}, n_points={self.n_points}",
        ]
        for model in self.mod_names:
            out.append(f" Model: {model}, rmse={self.sel(model=model).score():.3f}")
        return str.join("\n", out)

    @property
    def name(self) -> str:
        """name of comparer (=observation)"""
        return self.data.attrs["name"]

    @property
    def gtype(self):
        return self.data.attrs["gtype"]

    @property
    def variable_name(self) -> str:
        """name of variable"""
        return self.data.attrs["variable_name"]

    @property
    def n_points(self) -> int:
        """number of compared points"""
        return len(self.data[self._obs_name]) if self.data else 0

    @property
    def time(self) -> pd.DatetimeIndex:
        """time of compared data as pandas DatetimeIndex"""
        return self.data.time.to_index()

    @property
    def _mod_start(self) -> pd.Timestamp:
        mod_starts = [pd.Timestamp.max]
        for m in self.raw_mod_data.values():
            mod_starts.append(m.index[0])
        return min(mod_starts)

    @property
    def _mod_end(self) -> pd.Timestamp:
        mod_ends = [pd.Timestamp.min]
        for m in self.raw_mod_data.values():
            mod_ends.append(m.index[-1])
        return max(mod_ends)

    @property
    def start(self) -> pd.Timestamp:
        """start pd.Timestamp of compared data"""
        return self.time[0]

    @property
    def end(self) -> pd.Timestamp:
        """end pd.Timestamp of compared data"""
        return self.time[-1]

    @property
    def x(self):
        if "x" in self.data[self._obs_name].attrs.keys():
            return self.data[self._obs_name].attrs["x"]
        else:
            return self.data["x"].values

    @property
    def y(self):
        if "y" in self.data[self._obs_name].attrs.keys():
            return self.data[self._obs_name].attrs["y"]
        else:
            return self.data["y"].values

    @property
    def z(self):
        if "z" in self.data[self._obs_name].attrs.keys():
            return self.data[self._obs_name].attrs["z"]
        else:
            return self.data["z"].values

    @property
    def obs(self) -> np.ndarray:
        """Observation data as 1d numpy array"""
        return self.data[self._obs_name].to_dataframe().values

    @property
    def mod(self) -> np.ndarray:
        """Model data as 2d numpy array. Each column is a model"""
        return self.data[self.mod_names].to_dataframe().values

    @property
    def n_models(self) -> int:
        return len(self.mod_names)

    @property
    def mod_names(self) -> List[str]:
        return list(self.raw_mod_data.keys())

    @property
    def obs_name(self) -> str:
        """Name of observation (e.g. station name)"""
        return self._obs_name

    @property
    def weight(self) -> str:
        return self.data[self._obs_name].attrs["weight"]

    @property
    def _unit_text(self):
        warnings.warn("Use unit_text instead of _unit_text", FutureWarning)
        return self.unit_text

    @property
    def unit_text(self) -> str:
        """Variable name and unit as text suitable for plot labels"""
        return f"{self.data.attrs['variable_name']} [{self.data[self._obs_name].attrs['unit']}]"

    @property
    def metrics(self):
        return options.metrics.list

    @metrics.setter
    def metrics(self, values) -> None:
        if values is None:
            reset_option("metrics.list")
        else:
            options.metrics.list = _parse_metric(values, self.metrics)

    def _model_to_frame(self, mod_name: str) -> pd.DataFrame:
        """Convert single model data to pandas DataFrame"""

        df = self.data.to_dataframe().copy()
        other_models = [m for m in self.mod_names if m is not mod_name]
        df = df.drop(columns=other_models)
        df = df.rename(columns={mod_name: "mod_val", self._obs_name: "obs_val"})
        df["model"] = mod_name
        df["observation"] = self.name

        return df

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with all model data concatenated"""

        # TODO is this needed?, comment out for now
        # df = df.sort_index()
        df = pd.concat([self._model_to_frame(name) for name in self.mod_names])
        df["model"] = df["model"].astype("category")
        df["observation"] = df["observation"].astype("category")
        return df

    def __copy__(self):
        return deepcopy(self)

    def copy(self):
        return self.__copy__()

    def save(self, fn: Union[str, Path]) -> None:
        """Save to netcdf file

        Parameters
        ----------
        fn : str or Path
            filename
        """
        self.data.to_netcdf(fn)

    @staticmethod
    def load(fn: Union[str, Path]) -> "Comparer":
        """Load from netcdf file

        Parameters
        ----------
        fn : str or Path
            filename

        Returns
        -------
        Comparer
        """
        with xr.open_dataset(fn) as ds:
            data = ds.load()
        return Comparer(matched_data=data)

    def _to_observation(self) -> Observation:
        """Convert to Observation"""
        if self.gtype == "point":
            df = self.data[self._obs_name].to_dataframe()
            return PointObservation(
                data=df,
                name=self.name,
                x=self.x,
                y=self.y,
                z=self.z,
                # variable_name=self.variable_name,
                # units=self._unit_text,
            )
        elif self.gtype == "track":
            df = self.data[["x", "y", self._obs_name]].to_dataframe()
            return TrackObservation(
                data=df,
                item=2,
                x_item=0,
                y_item=1,
                name=self.name,
                # variable_name=self.variable_name,
                # units=self._unit_text,
            )
        else:
            raise NotImplementedError(f"Unknown gtype: {self.gtype}")

    def __add__(
        self, other: Union["Comparer", "ComparerCollection"]
    ) -> "ComparerCollection":

        from ._collection import ComparerCollection

        if not isinstance(other, (Comparer, ComparerCollection)):
            raise TypeError(f"Cannot add {type(other)} to {type(self)}")

        if isinstance(other, Comparer) and (self.name == other.name):
            assert type(self) == type(other), "Must be same type!"
            missing_models = set(self.mod_names) - set(other.mod_names)
            if len(missing_models) == 0:
                # same obs name and same model names
                cmp = self.copy()
                cmp.data = xr.concat([cmp.data, other.data], dim="time")
                # cc.data = cc.data[
                #    ~cc.data.time.to_index().duplicated(keep="last")
                # ]  # 'first'
                _, index = np.unique(cmp.data["time"], return_index=True)
                cmp.data = cmp.data.isel(time=index)

            else:
                cols = ["x", "y"]
                mod_data = [self.data[cols + [m]] for m in self.mod_names]
                for m in other.mod_names:
                    mod_data.append(other.data[cols + [m]])

                cls = self.__class__
                cmp = cls.__new__(cls)
                cmp.__init__(self._to_observation(), mod_data)
                # TODO cmp = cls.clone()

            return cmp
        else:
            cc = ComparerCollection()
            cc.add_comparer(self)
            cc.add_comparer(other)
            return cc

    def _model2obs_interp(
        self, obs, mod_df: pd.DataFrame, max_model_gap: Optional[TimeDeltaTypes]
    ):
        """interpolate model to measurement time"""
        df = _interp_time(mod_df.dropna(), obs.time)
        df[self._obs_name] = obs.values

        if max_model_gap is not None:
            df = _remove_model_gaps(df, mod_df.dropna().index, max_model_gap)

        return df

    def sel(
        self,
        model: IdOrNameTypes = None,
        start: TimeTypes = None,
        end: TimeTypes = None,
        time: TimeTypes = None,
        area: List[float] = None,
    ) -> "Comparer":
        """Select data based on model, time and/or area.

        Parameters
        ----------
        model : str or int or list of str or list of int, optional
            Model name or index. If None, all models are selected.
        start : str or datetime, optional
            Start time. If None, all times are selected.
        end : str or datetime, optional
            End time. If None, all times are selected.
        time : str or datetime, optional
            Time. If None, all times are selected.
        area : list of float, optional
            bbox: [x0, y0, x1, y1] or Polygon. If None, all areas are selected.

        Returns
        -------
        Comparer
            New Comparer with selected data.
        """
        d = self.data
        raw_mod_data = self.raw_mod_data
        if model is not None:
            models = [model] if np.isscalar(model) else model
            models = [_get_name(m, self.mod_names) for m in models]
            dropped_models = [m for m in self.mod_names if m not in models]
            d = d.drop_vars(dropped_models)
            raw_mod_data = {m: raw_mod_data[m] for m in models}
        if (start is not None) or (end is not None):
            if time is not None:
                raise ValueError("Cannot use both time and start/end")
            # TODO: can this be done without to_index? (simplify)
            d = d.sel(time=d.time.to_index().to_frame().loc[start:end].index)
        if time is not None:
            if (start is not None) or (end is not None):
                raise ValueError("Cannot use both time and start/end")
            d = d.sel(time=time)
        if area is not None:
            if _area_is_bbox(area):
                x0, y0, x1, y1 = area
                mask = (d.x > x0) & (d.x < x1) & (d.y > y0) & (d.y < y1)
            elif _area_is_polygon(area):
                polygon = np.array(area)
                xy = np.column_stack((d.x, d.y))
                mask = _inside_polygon(polygon, xy)
            else:
                raise ValueError("area supports bbox [x0,y0,x1,y1] and closed polygon")
            if self.gtype == "point":
                # if False, return empty data
                d = d if mask else d.isel(time=slice(None, 0))
            else:
                d = d.isel(time=mask)
        return self.__class__.from_compared_data(d, raw_mod_data)

    def where(
        self,
        cond: Union[bool, np.ndarray, xr.DataArray],
    ) -> "Comparer":
        """Return a new Comparer with values where cond is True

        Parameters
        ----------
        cond : bool, np.ndarray, xr.DataArray
            This selects the values to return.

        Returns
        -------
        Comparer
            New Comparer with values where cond is True and other otherwise.

        Examples
        --------
        >>> c2 = c.where(c.data.Observation > 0)
        """
        d = self.data.where(cond, other=np.nan)
        d = d.dropna(dim="time", how="all")
        return self.__class__.from_compared_data(d, self.raw_mod_data)

    def query(self, query: str) -> "Comparer":
        """Return a new Comparer with values where query cond is True

        Parameters
        ----------
        query : str
            Query string, see pandas.DataFrame.query

        Returns
        -------
        Comparer
            New Comparer with values where cond is True and other otherwise.

        Examples
        --------
        >>> c2 = c.query("Observation > 0")
        """
        d = self.data.query({"time": query})
        d = d.dropna(dim="time", how="all")
        return self.__class__.from_compared_data(d, self.raw_mod_data)

    def skill(
        self,
        by: Union[str, List[str]] = None,
        metrics: list = None,
        **kwargs,
    ) -> AggregatedSkill:
        """Skill assessment of model(s)

        Parameters
        ----------
        by : (str, List[str]), optional
            group by column name or by temporal bin via the freq-argument
            (using pandas pd.Grouper(freq)),
            e.g.: 'freq:M' = monthly; 'freq:D' daily
            by default ["model"]
        metrics : list, optional
            list of modelskill.metrics, by default modelskill.options.metrics.list
        freq : string, optional
            do temporal binning using pandas pd.Grouper(freq),
            typical examples: 'M' = monthly; 'D' daily
            by default None

        Returns
        -------
        AggregatedSkill
            skill assessment object

        See also
        --------
        sel
            a method for filtering/selecting data

        Examples
        --------
        >>> import modelskill as ms
        >>> cc = ms.compare(c2, mod)
        >>> cc['c2'].skill().round(2)
                       n  bias  rmse  urmse   mae    cc    si    r2
        observation
        c2           113 -0.00  0.35   0.35  0.29  0.97  0.12  0.99

        >>> cc['c2'].skill(by='freq:D').round(2)
                     n  bias  rmse  urmse   mae    cc    si    r2
        2017-10-27  72 -0.19  0.31   0.25  0.26  0.48  0.12  0.98
        2017-10-28   0   NaN   NaN    NaN   NaN   NaN   NaN   NaN
        2017-10-29  41  0.33  0.41   0.25  0.36  0.96  0.06  0.99
        """
        metrics = _parse_metric(metrics, self.metrics, return_list=True)

        # TODO remove in v1.1
        model, start, end, area = _get_deprecated_args(kwargs)

        cmp = self.sel(
            model=model,
            start=start,
            end=end,
            area=area,
        )
        if cmp.n_points == 0:
            raise ValueError("No data selected for skill assessment")

        by = _parse_groupby(by, cmp.n_models, n_obs=1, n_var=1)

        df = cmp.to_dataframe()  # TODO: avoid df if possible?
        res = _groupby_df(df.drop(columns=["x", "y"]), by, metrics)
        res = self._add_as_col_if_not_in_index(df, skilldf=res)
        return AggregatedSkill(res)

    def _add_as_col_if_not_in_index(self, df, skilldf):
        """Add a field to skilldf if unique in df"""
        FIELDS = ("observation", "model")

        for field in FIELDS:
            if (field == "model") and (self.n_models <= 1):
                continue
            if field not in skilldf.index.names:
                unames = df[field].unique()
                if len(unames) == 1:
                    skilldf.insert(loc=0, column=field, value=unames[0])
        return skilldf

    def score(
        self,
        metric=mtr.rmse,
        **kwargs,
    ) -> float:
        """Model skill score

        Parameters
        ----------
        metric : list, optional
            a single metric from modelskill.metrics, by default rmse

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
        >>> import modelskill as ms
        >>> cmp = ms.compare(c2, mod)
        >>> cmp.score()
        0.3517964910888918
        
        >>> cmp.score(metric=ms.metrics.mape)
        11.567399646108198
        """
        metric = _parse_metric(metric, self.metrics)
        if not (callable(metric) or isinstance(metric, str)):
            raise ValueError("metric must be a string or a function")

        # TODO remove in v1.1
        model, start, end, area = _get_deprecated_args(kwargs)

        s = self.skill(
            metrics=[metric],
            model=model,
            start=start,
            end=end,
            area=area,
        )
        if s is None:
            return
        df = s.df
        values = df[metric.__name__].values
        if len(values) == 1:
            values = values[0]
        return values

    def spatial_skill(
        self,
        bins=5,
        binsize: float = None,
        by: Union[str, List[str]] = None,
        metrics: list = None,
        n_min: int = None,
        **kwargs,
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
        >>> cmp = ms.compare(c2, mod)   # satellite altimeter vs. model
        >>> cmp.spatial_skill(metrics='bias')
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

        # TODO remove in v1.1
        model, start, end, area = _get_deprecated_args(kwargs)

        cmp = self.sel(
            model=model,
            start=start,
            end=end,
            area=area,
        )

        metrics = _parse_metric(metrics, self.metrics, return_list=True)
        if cmp.n_points == 0:
            raise ValueError("No data to compare")

        df = cmp.to_dataframe()
        df = _add_spatial_grid_to_df(df=df, bins=bins, binsize=binsize)

        # n_models = len(df.model.unique())
        # n_obs = len(df.observation.unique())

        # n_obs=1 because we only have one observation (**SingleObsComparer**)
        by = _parse_groupby(by=by, n_models=cmp.n_models, n_obs=1)
        if isinstance(by, str) or (not isinstance(by, Iterable)):
            by = [by]
        if "x" not in by:
            by.insert(0, "x")
        if "y" not in by:
            by.insert(0, "y")

        df = df.drop(columns=["x", "y"]).rename(columns=dict(xBin="x", yBin="y"))
        res = _groupby_df(df, by, metrics, n_min)
        return SpatialSkill(res.to_xarray().squeeze())

    @property
    def residual(self):
        return self.mod - np.vstack(self.obs)

    def remove_bias(self, correct="Model"):
        bias = self.residual.mean(axis=0)
        if correct == "Model":
            for j in range(self.n_models):
                mod_name = self.mod_names[j]
                mod_df = self.raw_mod_data[mod_name]
                mod_df[mod_name] = mod_df.values - bias[j]
                self.data[mod_name] = self.data[mod_name] - bias[j]
        elif correct == "Observation":
            # what if multiple models?
            self.data[self._obs_name] = self.obs + bias
        else:
            raise ValueError(
                f"Unknown correct={correct}. Only know 'Model' and 'Observation'"
            )
        return bias

    # TODO remove plotting methods in v1.1
    def scatter(
        self,
        *,
        bins=20,
        quantiles=None,
        fit_to_quantiles=False,
        show_points=None,
        show_hist=None,
        show_density=None,
        norm=None,
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

        warnings.warn(
            "This method is deprecated, use plot.scatter instead", FutureWarning
        )

        self.plot.scatter(
            bins=bins,
            quantiles=quantiles,
            fit_to_quantiles=fit_to_quantiles,
            show_points=show_points,
            show_hist=show_hist,
            show_density=show_density,
            norm=norm,
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

    def taylor(
        self,
        normalize_std=False,
        figsize=(7, 7),
        marker="o",
        marker_size=6.0,
        title="Taylor diagram",
        **kwargs,
    ):
        warnings.warn("taylor is deprecated, use plot.taylor instead", FutureWarning)

        self.plot.taylor(
            normalize_std=normalize_std,
            figsize=figsize,
            marker=marker,
            marker_size=marker_size,
            title=title,
            **kwargs,
        )

    def hist(
        self, *, model=None, bins=100, title=None, density=True, alpha=0.5, **kwargs
    ):
        warnings.warn("hist is deprecated. Use plot.hist instead.", FutureWarning)
        return self.plot.hist(model=model, bins=bins, title=title, density=density, alpha=alpha, **kwargs)

    def kde(self, ax=None, **kwargs) -> Axes:
        warnings.warn("kde is deprecated. Use plot.kde instead.", FutureWarning)

        return self.plot.kde(ax=ax, **kwargs)

    def plot_timeseries(
        self, title=None, *, ylim=None, figsize=None, backend="matplotlib", **kwargs
    ):
        warnings.warn(
            "plot_timeseries is deprecated. Use plot.timeseries instead.", FutureWarning
        )

        return self.plot.timeseries(
            title=title, ylim=ylim, figsize=figsize, backend=backend, **kwargs
        )

    def residual_hist(self, bins=100, title=None, color=None, **kwargs):

        warnings.warn(
            "residual_hist is deprecated. Use plot.residual_hist instead.",
            FutureWarning,
        )

        return self.plot.residual_hist(bins=bins, title=title, color=color, **kwargs)


class PointComparer(Comparer):
    pass


class TrackComparer(Comparer):
    pass
