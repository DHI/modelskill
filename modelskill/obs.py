"""
The `observation` module contains different types of Observation classes for
fixed locations (PointObservation), or locations moving in space (TrackObservation).

Examples
--------
>>> o1 = PointObservation("klagshamn.dfs0", item=0, x=366844, y=6154291, name="Klagshamn")
"""

from __future__ import annotations

from typing import Literal, Optional, Any, Union
import warnings
import pandas as pd
import xarray as xr

from .types import PointType, TrackType, GeometryType, DataInputType
from . import Quantity
from .timeseries import (
    TimeSeries,
    _parse_point_input,
    _parse_track_input,
)

# NetCDF attributes can only be str, int, float https://unidata.github.io/netcdf4-python/#attributes-in-a-netcdf-file
Serializable = Union[str, int, float]


def observation(
    data: DataInputType,
    *,
    gtype: Optional[Literal["point", "track"]] = None,
    **kwargs,
):
    """A factory function for creating an appropriate observation object
    based on the data and args.

    If 'x' or 'y' is given, a PointObservation is created.
    If 'x_item' or 'y_item' is given, a TrackObservation is created.

    Parameters
    ----------
    data : DataInputType
        The data to be used for creating the Observation object.
    gtype : Optional[Literal["point", "track"]]
        The geometry type of the data. If not specified, it will be guessed from the data.
    **kwargs
        Additional keyword arguments to be passed to the Observation constructor.

    Examples
    --------
    >>> import modelskill as ms
    >>> o_pt = ms.observation(df, item=0, x=366844, y=6154291, name="Klagshamn")
    >>> o_tr = ms.observation("lon_after_lat.dfs0", item="wl", x_item=1, y_item=0)
    """
    if gtype is None:
        geometry = _guess_gtype(**kwargs)
    else:
        geometry = GeometryType.from_string(gtype)

    return _obs_class_lookup[geometry](
        data=data,
        **kwargs,
    )


def _guess_gtype(**kwargs) -> GeometryType:
    """Guess geometry type from data"""

    if "x" in kwargs and "y" in kwargs:
        return GeometryType.POINT
    elif "x_item" in kwargs or "y_item" in kwargs:
        return GeometryType.TRACK
    else:
        warnings.warn(
            "Could not guess geometry type from data or args, assuming POINT geometry. Use PointObservation or TrackObservation to be explicit."
        )
        return GeometryType.POINT


def _validate_attrs(data_attrs: dict, attrs: Optional[dict]) -> None:
    # See similar method in xarray https://github.com/pydata/xarray/blob/main/xarray/backends/api.py#L165

    if attrs is None:
        return
    for k, v in attrs.items():
        if k in data_attrs:
            raise ValueError(f"attrs key {k} not allowed, conflicts with build-in key!")

        # TODO: check that v is a valid type for netcdf attributes, str, int, float
        if not isinstance(v, (str, int, float)):
            raise ValueError(
                f"attrs value {v} must be a valid type for netcdf attributes, str, int, float, not {type(v)}"
            )


class Observation(TimeSeries):
    def __init__(
        self,
        data: xr.Dataset,
        weight: float,
        color: str = "#d62728",  # TODO: cannot currently be set by user
        attrs: Optional[dict] = None,
    ) -> None:
        assert isinstance(data, xr.Dataset)

        data_var = str(list(data.data_vars)[0])
        data[data_var].attrs["kind"] = "observation"

        # check that user-defined attrs don't overwrite existing attrs!
        _validate_attrs(data.attrs, attrs)
        data.attrs = {**data.attrs, **(attrs or {})}
        data["time"] = self._parse_time(data.time)

        data_var = str(list(data.data_vars)[0])
        data[data_var].attrs["color"] = color
        super().__init__(data=data)
        self.data.attrs["weight"] = weight

    @property
    def attrs(self) -> dict[str, Any]:
        """Attributes of the observation"""
        return self.data.attrs

    @attrs.setter
    def attrs(self, value: dict[str, Serializable]) -> None:
        self.data.attrs = value

    @property
    def weight(self) -> float:
        """Weighting factor for skill scores"""
        return self.data.attrs["weight"]

    @weight.setter
    def weight(self, value: float) -> None:
        self.data.attrs["weight"] = value

    # TODO: move this to TimeSeries?
    @staticmethod
    def _parse_time(time):
        if isinstance(time, pd.DatetimeIndex):
            return time.dt.round("100us")
        else:
            return time  # can be RangeIndex


class PointObservation(Observation):
    """Class for observations of fixed locations

    Create a PointObservation from a dfs0 file or a pd.DataFrame.

    Parameters
    ----------
    data : str, Path, mikeio.Dataset, mikeio.DataArray, pd.DataFrame, pd.Series, xr.Dataset or xr.DataArray
        filename (.dfs0 or .nc) or object with the data
    item : (int, str), optional
        index or name of the wanted item/column, by default None
        if data contains more than one item, item must be given
    x : float, optional
        x-coordinate of the observation point, inferred from data if not given, else None
    y : float, optional
        y-coordinate of the observation point, inferred from data if not given, else None
    z : float, optional
        z-coordinate of the observation point, inferred from data if not given, else None
    name : str, optional
        user-defined name for easy identification in plots etc, by default file basename
    quantity : Quantity, optional
        The quantity of the observation, for validation with model results
        For MIKE dfs files this is inferred from the EUM information
    aux_items : list, optional
        list of names or indices of auxiliary items, by default None
    attrs : dict, optional
        additional attributes to be added to the data, by default None
    weight : float, optional
        weighting factor for skill scores, by default 1.0

    Examples
    --------
    >>> import modelskill as ms
    >>> o1 = ms.PointObservation("klagshamn.dfs0", item=0, x=366844, y=6154291, name="Klagshamn")
    >>> o2 = ms.PointObservation("klagshamn.dfs0", item="Water Level", x=366844, y=6154291)
    >>> o3 = ms.PointObservation(df, item=0, x=366844, y=6154291, name="Klagshamn")
    >>> o4 = ms.PointObservation(df["Water Level"], x=366844, y=6154291)
    """

    def __init__(
        self,
        data: PointType,
        *,
        item: Optional[int | str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        name: Optional[str] = None,
        weight: float = 1.0,
        quantity: Optional[Quantity] = None,
        aux_items: Optional[list[int | str]] = None,
        attrs: Optional[dict] = None,
    ) -> None:
        if not self._is_input_validated(data):
            data = _parse_point_input(
                data,
                name=name,
                item=item,
                quantity=quantity,
                aux_items=aux_items,
                x=x,
                y=y,
                z=z,
            )

        assert isinstance(data, xr.Dataset)
        super().__init__(data=data, weight=weight, attrs=attrs)

    @property
    def z(self):
        """z-coordinate of observation point"""
        return self._coordinate_values("z")

    @z.setter
    def z(self, value):
        self.data["z"] = value


class TrackObservation(Observation):
    """Class for observation with locations moving in space, e.g. satellite altimetry

    The data needs in addition to the datetime of each single observation point also, x and y coordinates.

    Create TrackObservation from dfs0 or DataFrame

    Parameters
    ----------
    data : (str, Path, mikeio.Dataset, pd.DataFrame, xr.Dataset)
        path to dfs0 file or object with track data
    item : (str, int), optional
        item name or index of values, by default None
        if data contains more than one item, item must be given
    name : str, optional
        user-defined name for easy identification in plots etc, by default file basename
    x_item : (str, int), optional
        item name or index of x-coordinate, by default 0
    y_item : (str, int), optional
        item name or index of y-coordinate, by default 1
    keep_duplicates : (str, bool), optional
        strategy for handling duplicate timestamps (xarray.Dataset.drop_duplicates):
        "first" to keep first occurrence, "last" to keep last occurrence,
        False to drop all duplicates, "offset" to add milliseconds to
        consecutive duplicates, by default "first"
    quantity : Quantity, optional
        The quantity of the observation, for validation with model results
        For MIKE dfs files this is inferred from the EUM information
    aux_items : list, optional
        list of names or indices of auxiliary items, by default None
    attrs : dict, optional
        additional attributes to be added to the data, by default None
    weight : float, optional
        weighting factor for skill scores, by default 1.0

    Examples
    --------
    >>> import modelskill as ms
    >>> o1 = ms.TrackObservation("track.dfs0", item=2, name="c2")

    >>> o1 = ms.TrackObservation("track.dfs0", item="wind_speed", name="c2")

    >>> o1 = ms.TrackObservation("lon_after_lat.dfs0", item="wl", x_item=1, y_item=0)

    >>> o1 = ms.TrackObservation("track_wl.dfs0", item="wl", x_item="lon", y_item="lat")

    >>> df = pd.DataFrame(
    ...         {
    ...             "t": pd.date_range("2010-01-01", freq="10s", periods=n),
    ...             "x": np.linspace(0, 10, n),
    ...             "y": np.linspace(45000, 45100, n),
    ...             "swh": [0.1, 0.3, 0.4, 0.5, 0.3],
    ...         }
    ... )
    >>> df = df.set_index("t")
    >>> df
                        x        y  swh
    t
    2010-01-01 00:00:00   0.0  45000.0  0.1
    2010-01-01 00:00:10   2.5  45025.0  0.3
    2010-01-01 00:00:20   5.0  45050.0  0.4
    2010-01-01 00:00:30   7.5  45075.0  0.5
    2010-01-01 00:00:40  10.0  45100.0  0.3
    >>> t1 = TrackObservation(df, name="fake")
    >>> t1.n_points
    5
    >>> t1.values
    array([0.1, 0.3, 0.4, 0.5, 0.3])
    >>> t1.time
    DatetimeIndex(['2010-01-01 00:00:00', '2010-01-01 00:00:10',
               '2010-01-01 00:00:20', '2010-01-01 00:00:30',
               '2010-01-01 00:00:40'],
              dtype='datetime64[ns]', name='t', freq=None)
    >>> t1.x
    array([ 0. ,  2.5,  5. ,  7.5, 10. ])
    >>> t1.y
    array([45000., 45025., 45050., 45075., 45100.])

    """

    def __init__(
        self,
        data: TrackType,
        *,
        item: Optional[int | str] = None,
        name: Optional[str] = None,
        weight: float = 1.0,
        x_item: Optional[int | str] = 0,
        y_item: Optional[int | str] = 1,
        keep_duplicates: bool | str = "first",
        offset_duplicates: float = 0.001,
        quantity: Optional[Quantity] = None,
        aux_items: Optional[list[int | str]] = None,
        attrs: Optional[dict] = None,
    ) -> None:
        if not self._is_input_validated(data):
            if offset_duplicates != 0.001:
                warnings.warn(
                    "The 'offset_duplicates' argument is deprecated, use 'keep_duplicates' argument.",
                    FutureWarning,
                )
            data = _parse_track_input(
                data=data,
                name=name,
                item=item,
                quantity=quantity,
                x_item=x_item,
                y_item=y_item,
                keep_duplicates=keep_duplicates,
                offset_duplicates=offset_duplicates,
                aux_items=aux_items,
            )
        assert isinstance(data, xr.Dataset)
        super().__init__(data=data, weight=weight, attrs=attrs)


def unit_display_name(name: str) -> str:
    """Display name

    Examples
    --------
    >>> unit_display_name("meter")
    m
    """

    res = (
        name.replace("meter", "m")
        .replace("_per_", "/")
        .replace(" per ", "/")
        .replace("second", "s")
        .replace("sec", "s")
        .replace("degree", "°")
    )

    return res


_obs_class_lookup = {
    GeometryType.POINT: PointObservation,
    GeometryType.TRACK: TrackObservation,
}
