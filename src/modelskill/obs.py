"""
# Observations

ModelSkill supports five types of observations:

* [`PointObservation`](`modelskill.PointObservation`) - a point timeseries from a dfs0/nc file or a DataFrame
* [`TrackObservation`](`modelskill.TrackObservation`) - a track (moving point) timeseries from a dfs0/nc file or a DataFrame
* [`VerticalObservation`](`modelskill.VerticalObservation`) - a vertical profile from a dfs0/nc file or a DataFrame
* [`NodeObservation`](`modelskill.NodeObservation`) - a network node timeseries for a named node or break point.
* [`ReachObservation`](`modelskill.ReachObservation`) - a network reach timeseries for a quantity uniform along the reach.

An observation can be created by explicitly invoking one of the above classes or using the [`observation()`](`modelskill.observation`) function which will return the appropriate type based on the input data (if possible).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Literal,
    NamedTuple,
    Sequence,
    Union,
    overload,
)
from typing_extensions import Self
import warnings
import numpy as np
import pandas as pd
import xarray as xr

from .types import PointType, TrackType, VerticalType, GeometryType, DataInputType
from . import Quantity
from .timeseries import (
    TimeSeries,
    _parse_xyz_point_input,
    _parse_track_input,
    _parse_vertical_input,
    _parse_network_node_input,
    _parse_network_breakpoint_input,
)


# NetCDF attributes can only be str, int, float https://unidata.github.io/netcdf4-python/#attributes-in-a-netcdf-file
Serializable = Union[str, int, float]

# Where a node observation sits: the name the network gave the node, or a
# breakpoint given as (reach_id, distance) along a reach.
NodeLocation = Union[str, tuple[str, float]]


def observation(
    data: DataInputType,
    *,
    gtype: Literal["point", "track", "vertical", "node", "reach"] | None = None,
    **kwargs,
) -> (
    PointObservation
    | TrackObservation
    | VerticalObservation
    | NodeObservation
    | ReachObservation
):
    """Create an appropriate observation object.

    A factory function for creating an appropriate observation object
    based on the data and args.

    If 'x' or 'y' is given, a PointObservation is created.
    If 'x_item' or 'y_item' is given, a TrackObservation is created.
    If 'z_item' is given, a VerticalObservation is created.
    If 'at' is given, a NodeObservation is created.
    If 'reach' is given, a ReachObservation is created.
    If gtype is explicitly given, it will be used to determine the type of observation.

    Parameters
    ----------
    data : DataInputType
        The data to be used for creating the Observation object.
    gtype : Literal["point", "track", "vertical", "node", "reach"] | None
        The geometry type of the data. If not specified, it will be guessed from the data.
    **kwargs
        Additional keyword arguments to be passed to the Observation constructor.

    Returns
    -------
    PointObservation or TrackObservation or VerticalObservation or NodeObservation or ReachObservation
        An observation object of the appropriate type

    Examples
    --------
    >>> import modelskill as ms
    >>> o_pt = ms.observation(df, item=0, x=366844, y=6154291, name="Klagshamn")
    >>> o_tr = ms.observation("lon_after_lat.dfs0", item="wl", x_item=1, y_item=0)
    >>> o_node = ms.observation(df, item="Water Level", at="123", name="123")
    >>> o_reach = ms.observation(df, item="Discharge", reach="reach_1", name="reach_1_Q")
    """
    if gtype is None:
        geometry = _guess_gtype(**kwargs)
    else:
        geometry = GeometryType(gtype)

    return _obs_class_lookup[geometry](
        data=data,
        **kwargs,
    )


def _guess_gtype(**kwargs) -> GeometryType:
    """Guess geometry type from data"""

    if "z_item" in kwargs:
        return GeometryType.VERTICAL
    elif "x" in kwargs and "y" in kwargs:
        return GeometryType.POINT
    elif "x_item" in kwargs or "y_item" in kwargs:
        return GeometryType.TRACK
    elif "at" in kwargs:
        return GeometryType.NODE
    elif "reach" in kwargs:
        return GeometryType.REACH
    else:
        warnings.warn(
            "Could not guess geometry type from data or args, assuming POINT geometry. "
            "Use PointObservation, TrackObservation, VerticalObservation, NodeObservation, ReachObservation to be explicit."
        )
        return GeometryType.POINT


def _item_names(data: Any) -> list[str]:
    """Names of the individual timeseries held by an already-opened data source."""
    if isinstance(data, pd.DataFrame):
        return [str(c) for c in data.columns]
    if isinstance(data, xr.Dataset):
        return [str(v) for v in data.data_vars]
    if hasattr(data, "names"):  # mikeio.Dataset
        return [str(n) for n in data.names]
    if hasattr(data, "name"):  # pd.Series, mikeio.DataArray, xr.DataArray
        return [str(data.name)]
    raise ValueError(
        f"Cannot determine item names from data of type {type(data).__name__}"
    )


class _Station(NamedTuple):
    """One measured timeseries, resolved to the network location it belongs to."""

    item_name: str  #: name of the item in the data source
    name: str  #: display name for the observation
    location: str | tuple[str, float]  #: node name, or (reach, chainage)
    kind: Literal["node", "reach"]  #: which observation class fits
    quantity: str  #: modelled quantity name


class _MikePlusStationResolver:
    """Resolves data source items to network locations, via a MIKE+ database.

    A MIKE+ project ships a sqlite database alongside its result files. Two of
    its tables say where the measured timeseries belong in the network:

    * ``m_Measurement`` - one row per measured timeseries, naming the file
      (``tsfilename``) and the item within it (``tsitemname``), plus the
      modelled quantity (``resitemname``).
    * ``m_Station`` - the location, as ``locationid`` plus a ``locationtype``
      saying whether that identifier names a node or a link.

    Everything MIKE+ specific is contained here - the table names, the join, the
    ``locationtype`` codes, the encoding of ``resitemname`` - so a change to the
    database layout is a change to this class alone. Callers see only
    :class:`_Station`.
    """

    _TABLES: dict[str, set[str]] = {
        "m_Station": {
            "muid",
            "locationid",
            "locationtype",
            "chainagevalue",
            "assetname",
        },
        "m_Measurement": {
            "measurementstationid",
            "tsfilename",
            "tsitemname",
            "resitemname",
        },
    }

    # m_Station.locationtype codes. 8 is a junction and 12 a tank or reservoir;
    # both are graph nodes. 9 is a link, which becomes a breakpoint when the
    # station carries a chainage and a whole reach when it does not. Unknown
    # codes raise rather than guess.
    _NODE_TYPES = frozenset({8, 12})
    _LINK_TYPES = frozenset({9})

    _QUERY = """
        SELECT m.tsitemname    AS item_name,
               m.tsfilename    AS tsfilename,
               m.resitemname   AS resitemname,
               s.assetname     AS assetname,
               s.locationid    AS locationid,
               s.locationtype  AS locationtype,
               s.chainagevalue AS chainagevalue
        FROM m_Measurement m
        JOIN m_Station s ON s.muid = m.measurementstationid
    """

    def __init__(
        self,
        db: str | Path | sqlite3.Connection,
        *,
        source: str | None = None,
    ) -> None:
        """Read the join, and the station names needed to explain a failure.

        ``db`` is a path or an already-open connection; an open one is left open.
        ``source`` restricts the measurements to one result file, matched on file
        name, so a full path is fine. Without it, measurements from every file
        are considered and an item registered against two of them raises.
        """
        self._source = source

        if isinstance(db, sqlite3.Connection):
            conn, opened = db, None
        else:
            conn = opened = sqlite3.connect(str(db))
        try:
            self._validate(conn)

            query, params = self._QUERY, []
            if source is not None:
                query += " WHERE m.tsfilename LIKE ?"
                params.append(f"%{Path(source).name}%")
            rows = pd.read_sql_query(query, conn, params=params)

            # Read the station names now rather than on demand: the only other
            # use is naming stations that carry no measurement, on the failure
            # path, and reading them here is what lets the connection close.
            self._assets = set(
                pd.read_sql_query("SELECT assetname FROM m_Station", conn)["assetname"]
                .dropna()
                .tolist()
            )
        finally:
            if opened is not None:
                opened.close()

        rows["quantity"] = rows["resitemname"].str.split(";").str[0].str.strip()
        self._rows = rows

    def resolve(
        self,
        item_names: Iterable[str],
        *,
        quantity: str | None = None,
        kind: Literal["node", "reach"] | None = None,
        on_missing: Literal["raise", "skip"] = "raise",
    ) -> list[_Station]:
        """Resolve item names, e.g. the columns of a dfs0, to their locations.

        ``quantity`` selects one of several measured quantities; left None it is
        inferred, and raises when the selection holds more than one. ``kind``
        restricts the result to nodes or to reaches. ``on_missing="skip"`` drops
        items the database does not register, which otherwise raise.
        """
        requested = list(dict.fromkeys(item_names))
        rows = self._rows[self._rows["item_name"].isin(requested)].copy()

        missing = [item for item in requested if item not in set(rows["item_name"])]
        if missing and on_missing == "raise":
            raise ValueError(
                f"{len(missing)} of {len(requested)} items could not be resolved "
                f"against the MIKE+ database.\n"
                + self._unresolved_message(missing)
                + '\n  Pass on_missing="skip" to ignore these.'
            )

        if ambiguous := sorted(
            rows.loc[rows.duplicated("item_name", keep=False), "item_name"].unique()
        ):
            raise ValueError(
                f"Item(s) {ambiguous} are registered against more than one file. "
                "Pass 'source' to say which file the data comes from."
            )

        if rows.empty:
            raise ValueError("No items could be resolved against the MIKE+ database.")

        located = rows.apply(self._location, axis=1)
        rows["kind"] = [k for k, _ in located]
        rows["location"] = [location for _, location in located]

        if quantity is None:
            pool = rows if kind is None else rows[rows["kind"] == kind]
            available = sorted(pool["quantity"].unique())
            if len(available) == 0:
                raise ValueError(
                    f"No {kind} locations found. Quantities present: "
                    f"{rows['quantity'].value_counts().to_dict()}."
                )
            if len(available) > 1:
                raise ValueError(
                    "Several quantities present, so 'quantity' cannot be inferred: "
                    f"{pool['quantity'].value_counts().to_dict()}. "
                    f"Pass one of {available}."
                )
            quantity = available[0]

        selection = rows[rows["quantity"] == quantity]
        if selection.empty:
            raise ValueError(
                f"Quantity {quantity!r} not found. Available: "
                f"{rows['quantity'].value_counts().to_dict()}."
            )

        if kind is not None:
            of_kind = selection[selection["kind"] == kind]
            if of_kind.empty:
                other = sorted(selection["kind"].unique())
                raise ValueError(
                    f"All {len(selection)} {quantity!r} station(s) are of kind "
                    f"{other}, not {kind!r}."
                )
            selection = of_kind

        names = self._display_names(selection)
        return [
            _Station(
                item_name=str(row.item_name),
                name=str(name),
                location=row.location,
                kind=row.kind,
                quantity=str(row.quantity),
            )
            for name, row in zip(names, selection.itertuples())
        ]

    def _validate(self, conn: sqlite3.Connection) -> None:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if missing := sorted(set(self._TABLES) - tables):
            raise ValueError(
                f"Database is missing table(s) {missing}. "
                "A MIKE+ database with 'm_Station' and 'm_Measurement' is required."
            )
        for table, required in self._TABLES.items():
            columns = {row[1] for row in conn.execute(f"PRAGMA table_info([{table}])")}
            if missing_cols := sorted(required - columns):
                raise ValueError(
                    f"Table '{table}' is missing column(s) {missing_cols}. "
                    "The database layout is not the one modelskill expects."
                )

    def _location(self, row: pd.Series) -> tuple[str, str | tuple[str, float]]:
        # A link station with a chainage names a point along a reach, which is a
        # node observation at a breakpoint. Without a chainage it names the reach
        # as a whole.
        try:
            location_type = int(row["locationtype"])
        except (TypeError, ValueError):
            location_type = -1

        location_id = str(row["locationid"])
        if location_type in self._NODE_TYPES:
            return "node", location_id
        if location_type in self._LINK_TYPES:
            chainage = row["chainagevalue"]
            if pd.isna(chainage):
                return "reach", location_id
            return "node", (location_id, float(chainage))

        raise ValueError(
            f"Station '{row['locationid']}' has unsupported locationtype "
            f"{row['locationtype']!r}. Known codes are "
            f"{sorted(self._NODE_TYPES | self._LINK_TYPES)}."
        )

    def _unresolved_message(self, missing: Sequence[str]) -> str:
        known = [item for item in missing if item in self._assets]
        unknown = [item for item in missing if item not in self._assets]

        lines = []
        if known:
            where = f" for '{Path(self._source).name}'" if self._source else ""
            lines.append(
                f"  Known station, no measurement registered{where} ({len(known)}):\n"
                + "\n".join(f"    {item}" for item in known)
            )
        if unknown:
            lines.append(
                f"  Not found in the database ({len(unknown)}):\n"
                + "\n".join(f"    {item}" for item in unknown)
            )
        return "\n".join(lines)

    @staticmethod
    def _display_names(selection: pd.DataFrame) -> pd.Series:
        # assetname is far shorter than the raw item name and is normally unique,
        # but it is only safe as a display name when it distinguishes every row.
        assets = selection["assetname"]
        if assets.notna().all() and assets.nunique() == len(selection):
            return assets.astype(str)
        return selection["item_name"].astype(str)


def _observations_from_mikeplus(
    cls: type,
    *,
    data: PointType,
    db: Any,
    kind: Literal["node", "reach"],
    location_arg: str,
    quantity: Quantity | str | None,
    source: str | None,
    on_missing: Literal["raise", "skip"],
    aux_items: list[int | str] | None,
    attrs: dict | None,
) -> list[Any]:
    """Build observations from a data source and a MIKE+ database."""
    from .timeseries._point import _open_and_name

    if source is None and isinstance(data, (str, Path)):
        source = str(data)

    # Open once rather than per observation; a path would otherwise be re-read
    # for every station in the database.
    opened, _ = _open_and_name(data, None)

    given_quantity = quantity if isinstance(quantity, Quantity) else None
    wanted = quantity.name if isinstance(quantity, Quantity) else quantity

    stations = _MikePlusStationResolver(db, source=source).resolve(
        _item_names(opened),
        quantity=wanted,
        kind=kind,
        on_missing=on_missing,
    )

    observations = []
    for station in stations:
        obs = cls(
            opened,
            item=station.item_name,
            name=station.name,
            quantity=given_quantity,
            aux_items=aux_items,
            attrs=attrs,
            **{location_arg: station.location},
        )
        if given_quantity is None:
            # The database names the quantity; the data source knows its unit.
            obs.quantity = Quantity(
                name=station.quantity,
                unit=obs.quantity.unit,
                is_directional=obs.quantity.is_directional,
            )
        observations.append(obs)
    return observations


def _validate_attrs(data_attrs: dict, attrs: dict | None) -> None:
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
        attrs: dict | None = None,
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
        item: int | str | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        name: str | None = None,
        weight: float = 1.0,
        quantity: Quantity | None = None,
        aux_items: list[int | str] | None = None,
        attrs: dict | None = None,
    ) -> None:
        if not self._is_input_validated(data):
            data = _parse_xyz_point_input(
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
        item: int | str | None = None,
        name: str | None = None,
        weight: float = 1.0,
        x_item: int | str | None = 0,
        y_item: int | str | None = 1,
        keep_duplicates: Literal["first", "last", False] = "first",
        quantity: Quantity | None = None,
        aux_items: list[int | str] | None = None,
        attrs: dict | None = None,
    ) -> None:
        if not self._is_input_validated(data):
            data = _parse_track_input(
                data=data,
                name=name,
                item=item,
                quantity=quantity,
                x_item=x_item,
                y_item=y_item,
                keep_duplicates=keep_duplicates,
                aux_items=aux_items,
            )
        assert isinstance(data, xr.Dataset)
        super().__init__(data=data, weight=weight, attrs=attrs)


class VerticalObservation(Observation):
    """Class for observations of vertical profiles.

    Create a VerticalObservation from a dfs0/nc file or tabular data
    containing time, vertical coordinate, and observed values.

    Parameters
    ----------
    data : (str, Path, pd.DataFrame, mikeio.Dfs0, mikeio.Dataset, xr.Dataset)
        Input data with vertical profile observations.
    item : int or str, optional
        Index or name of the primary observation item.
        If the input contains more than one candidate value item,
        this argument must be provided.
    x : float, optional
        x-coordinate of the observation location. If not provided,
        it is inferred from data when possible.
    y : float, optional
        y-coordinate of the observation location. If not provided,
        it is inferred from data when possible.
    z_item : int or str, optional
        Index or name of the vertical coordinate item, by default 0.
    name : str, optional
        User-defined name for identification in plots and summaries.
    weight : float, optional
        Weighting factor for skill scores, by default 1.0.
    quantity : Quantity, optional
        Physical quantity metadata used for validation against model results.
    aux_items : list[int | str], optional
        List of auxiliary item names or indices to keep in the dataset.
    attrs : dict, optional
        Additional attributes to be added to the underlying dataset.

    Examples
    --------
    >>> import modelskill as ms
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "z": [0.0, -5.0, -10.0, 0.0, -5.0, -10.0],
    ...         "value": [0.1, 0.3, 0.4, 0.5, 0.3, 0.3],
    ...     },
    ...     index=pd.to_datetime(
    ...         [
    ...             "2010-01-01 01:00:00",
    ...             "2010-01-01 01:00:00",
    ...             "2010-01-01 01:00:00",
    ...             "2010-01-01 02:00:00",
    ...             "2010-01-01 02:00:00",
    ...             "2010-01-01 02:00:00",
    ...         ]
    ...     ),
    ... )
    >>> df.index.name = "t"
    >>> print(df.to_string())
                           z  value
    t
    2010-01-01 01:00:00   0.0    0.1
    2010-01-01 01:00:00  -5.0    0.3
    2010-01-01 01:00:00 -10.0    0.4
    2010-01-01 02:00:00   0.0    0.5
    2010-01-01 02:00:00  -5.0    0.3
    2010-01-01 02:00:00 -10.0    0.3

    >>> o = ms.VerticalObservation(df, item="value", z_item="z", x=12.0, y=55.0)
    """

    def __init__(
        self,
        data: VerticalType,
        *,
        item: int | str | None = None,
        x: float | None = None,
        y: float | None = None,
        z_item: int | str | None = 0,
        name: str | None = None,
        weight: float = 1.0,
        quantity: Quantity | None = None,
        aux_items: list[int | str] | None = None,
        attrs: dict | None = None,
    ) -> None:
        if not self._is_input_validated(data):
            data = _parse_vertical_input(
                data,
                name=name,
                item=item,
                quantity=quantity,
                aux_items=aux_items,
                z_item=z_item,
                x=x,
                y=y,
            )
        assert isinstance(data, xr.Dataset)
        super().__init__(data=data, weight=weight, attrs=attrs)

    @property
    def z(self):
        return self._coordinate_values("z")


class NodeObservation(Observation):
    """Class for observations at network nodes.

    Create a NodeObservation from a DataFrame or other data source.
    The ``at`` parameter accepts two forms:

    * **str** — the node's name in the model (e.g. a Res1D node name).
    * **tuple[str, float]** — a breakpoint, as ``(reach_id, distance)`` along a
      reach.

    Both are resolved against the network when the observation is matched.

    .. note::
        "Node" in this API follows the broad graph sense: it covers both
        junctions (named connection points) and chainage points (breakpoints
        along a reach).  MIKE 1D users who distinguish *node* (junction) from
        *gridpoint* sharply can use the ``(reach_id, distance)`` tuple form to
        target a specific chainage point, or :class:`ReachObservation` when
        the quantity is uniform across the whole reach and any breakpoint will do.

    To create multiple NodeObservation objects from a single data source,
    use :meth:`from_multiple`.

    Parameters
    ----------
    data : str, Path, mikeio.Dataset, mikeio.DataArray, pd.DataFrame, pd.Series, xr.Dataset or xr.DataArray
        data source with time series for the node
    at : str or tuple[str, float]
        Observation location. Accepted forms:

        * **str** — the node's name in the model (e.g. a Res1D node name).
        * **tuple[str, float]** — a breakpoint, as ``(reach_id, distance)``.
    item : (int, str), optional
        index or name of the wanted item/column, by default None
        if data contains more than one item, item must be given
    name : str, optional
        user-defined name for easy identification in plots etc, by default derived from data
    weight : float, optional
        weighting factor for skill scores, by default 1.0
    quantity : Quantity, optional
        The quantity of the observation, for validation with model results
    aux_items : list, optional
        list of names or indices of auxiliary items, by default None
    attrs : dict, optional
        additional attributes to be added to the data, by default None

    Examples
    --------
    >>> import modelskill as ms
    >>> o1 = ms.NodeObservation(data, at="123", name="123")
    >>> o2 = ms.NodeObservation(df, item="Water Level", at="456")
    >>>
    >>> # String alias resolved at match time
    >>> o3 = ms.NodeObservation(data, at="node_A")
    >>>
    >>> # Breakpoint as (reach_id, distance) tuple
    >>> o4 = ms.NodeObservation(data, at=("reach_1", 24.5))
    >>>
    >>> # Multiple node observations from separate data sources
    >>> obs = ms.NodeObservation.from_multiple(nodes={"123": df1, "456": df2})
    """

    def __init__(
        self,
        data: PointType,
        *,
        at: str | tuple[str, float],
        item: int | str | None = None,
        name: str | None = None,
        weight: float = 1.0,
        quantity: Quantity | None = None,
        aux_items: list[int | str] | None = None,
        attrs: dict | None = None,
    ) -> None:
        if isinstance(at, (int, np.integer)) and not isinstance(at, bool):
            raise TypeError(
                "'at' takes a node name or a (reach, distance) pair, not an integer. "
                "The integers a Network hands out are an internal index; "
                "network.recall(<int>) gives the name back."
            )
        if isinstance(at, tuple):
            reach, distance = str(at[0]), float(at[1])
            if not self._is_input_validated(data):
                data = _parse_network_breakpoint_input(
                    data,
                    name=name,
                    item=item,
                    quantity=quantity,
                    aux_items=aux_items,
                    reach=reach,
                    distance=distance,
                )
        else:
            if not self._is_input_validated(data):
                data = _parse_network_node_input(
                    data,
                    name=name,
                    item=item,
                    quantity=quantity,
                    node=at,
                    aux_items=aux_items,
                )
        assert isinstance(data, xr.Dataset)
        super().__init__(data=data, weight=weight, attrs=attrs)

    @property
    def at(self) -> str | tuple[str, float]:
        """Observation location: a node name, or a ``(reach_id, distance)`` breakpoint."""
        if "reach" in self.data.coords:
            return (
                str(self.data.coords["reach"].item()),
                float(self.data.coords["distance"].item()),
            )
        return str(self.data.coords["node"].item())

    def _create_new_instance(self, data: xr.Dataset) -> Self:
        """Reconstruct instance from a dataset slice."""
        if "reach" in data.coords:
            return self.__class__(
                data,
                at=(
                    str(data.coords["reach"].item()),
                    float(data.coords["distance"].item()),
                ),
            )
        return self.__class__(data, at=str(data.coords["node"].item()))

    @overload
    @classmethod
    def from_multiple(
        cls,
        *,
        data: PointType,
        nodes: dict[NodeLocation, str | int],
        quantity: Quantity | None = None,
        aux_items: list[int | str] | None = None,
        attrs: dict | None = None,
    ) -> list[NodeObservation]: ...

    @overload
    @classmethod
    def from_multiple(
        cls,
        *,
        nodes: dict[NodeLocation, PointType],
        quantity: Quantity | None = None,
        aux_items: list[int | str] | None = None,
        attrs: dict | None = None,
    ) -> list[NodeObservation]:
        pass

    @overload
    @classmethod
    def from_multiple(
        cls,
        *,
        data: PointType,
        db: str | Path | Any,
        quantity: Quantity | str | None = None,
        source: str | None = None,
        on_missing: Literal["raise", "skip"] = "raise",
        aux_items: list[int | str] | None = None,
        attrs: dict | None = None,
    ) -> list[NodeObservation]: ...

    @classmethod
    def from_multiple(
        cls,
        *,
        data: PointType | None = None,
        nodes: dict[NodeLocation, Any] | None = None,
        db: str | Path | Any | None = None,
        quantity: Quantity | str | None = None,
        source: str | None = None,
        on_missing: Literal["raise", "skip"] = "raise",
        aux_items: list[int | str] | None = None,
        attrs: dict | None = None,
    ) -> list[NodeObservation]:
        """Create multiple NodeObservation objects.

        Two calling conventions are supported:

        1. **Separate data sources** — pass only ``nodes`` as a dict mapping
           each node ID to its own data source (file path, DataFrame, etc.)::

               obs = NodeObservation.from_multiple(nodes={"123": df1, "456": "sensor.csv"})

        2. **Shared data source** — pass a single ``data`` object together with
           ``nodes`` as a dict mapping each node ID to the column name or index
           to select from ``data``::

               obs = NodeObservation.from_multiple(data=df, nodes={"123": "col_a", "456": "col_b"})

        3. **MIKE+ database** — pass a single ``data`` object together with
           ``db``, and the locations are looked up in the database::

               obs = NodeObservation.from_multiple(data="calib.dfs0", db="model.sqlite")

           One observation is created per item of ``data`` that the database
           places on a node, so several sensors at the same node are all kept.

        Parameters
        ----------
        data : PointType, optional
            Shared data source (required when ``nodes`` values are column
            selectors, and when ``db`` is given).
        nodes : dict[str | tuple[str, float], PointType | str | int]
            Mapping of location -> data source or column selector. A location
            takes either of the forms accepted by ``at``: a node name, or a
            ``(reach_id, distance)`` breakpoint.

            Note that a location can appear only once, so this form cannot
            express several observations at the same node. Use ``db`` when the
            data has several sensors at one location.
        db : str, Path or sqlite3.Connection, optional
            MIKE+ database locating the items of ``data`` in the network.
            Mutually exclusive with ``nodes``.
        quantity : Quantity or str, optional
            Physical quantity metadata, by default None. With ``db``, a string
            selects which quantity to build observations for and the metadata
            comes from the database; omit it and the quantity is inferred when
            the data holds only one.
        source : str, optional
            With ``db``, the file the items come from. Taken from ``data`` when
            that is a path, by default None.
        on_missing : {"raise", "skip"}, optional
            With ``db``, what to do with items the database cannot place, by
            default "raise".
        aux_items : list[int | str] | None, optional
            Auxiliary items, by default None.
        attrs : dict | None, optional
            Additional attributes, by default None.

        Returns
        -------
        list[NodeObservation]
            List of NodeObservation objects.

        Raises
        ------
        ValueError
            If both ``nodes`` and ``db`` are given, if neither is, or if the
            database cannot resolve the requested items.
        """
        if db is not None:
            if nodes is not None:
                raise ValueError(
                    "'nodes' and 'db' are mutually exclusive: the database "
                    "supplies the locations."
                )
            if data is None:
                raise ValueError("'data' is required when 'db' is given")
            return _observations_from_mikeplus(
                cls,
                data=data,
                db=db,
                kind="node",
                location_arg="at",
                quantity=quantity,
                source=source,
                on_missing=on_missing,
                aux_items=aux_items,
                attrs=attrs,
            )

        if isinstance(quantity, str):
            raise TypeError(
                "'quantity' must be a Quantity unless 'db' is given, got str"
            )

        if nodes is None:
            raise ValueError("'nodes' argument is required")
        if not isinstance(nodes, dict):
            raise TypeError(
                f"'nodes' must be a dict mapping node_id -> data_source, got {type(nodes).__name__}"
            )

        node_ids = list(nodes.keys())

        if data is None:
            data_sources: list[PointType] = list(nodes.values())  # type: ignore[list-item]
            return [
                cls(
                    data_i,
                    at=node_i,
                    item=None,
                    quantity=quantity,
                    aux_items=aux_items,
                    attrs=attrs,
                )
                for data_i, node_i in zip(data_sources, node_ids)
            ]
        else:
            node_items: list[int | str | None] = list(nodes.values())  # type: ignore[list-item]
            return [
                cls(
                    data,
                    at=node_i,
                    item=item_i,
                    quantity=quantity,
                    aux_items=aux_items,
                    attrs=attrs,
                )
                for node_i, item_i in zip(node_ids, node_items)
            ]


class ReachObservation(Observation):
    """Class for observations representing a quantity uniform across a network reach.

    Some quantities (e.g. discharge in a river reach) are constant for the
    whole reach, even though the underlying model stores values at
    nodes/breakpoints.  A ReachObservation associates a timeseries with a
    named reach; when matched against a
    :class:`~modelskill.model.network.NetworkModelResult` the data is
    extracted from an arbitrary breakpoint that belongs to that reach.

    Parameters
    ----------
    data : str, Path, mikeio.Dataset, mikeio.DataArray, pd.DataFrame, pd.Series, xr.Dataset or xr.DataArray
        data source with time series for the reach quantity
    reach : str
        Reach identifier (reach name / reach ID) in the network.
    item : (int, str), optional
        index or name of the wanted item/column, by default None
        if data contains more than one item, item must be given
    name : str, optional
        user-defined name for easy identification in plots etc, by default derived from data
    weight : float, optional
        weighting factor for skill scores, by default 1.0
    quantity : Quantity, optional
        The quantity of the observation, for validation with model results
    aux_items : list, optional
        list of names or indices of auxiliary items, by default None
    attrs : dict, optional
        additional attributes to be added to the data, by default None

    Examples
    --------
    >>> import modelskill as ms
    >>> o1 = ms.ReachObservation(df, reach="reach_1", name="Q_reach_1")
    >>> o2 = ms.ReachObservation(df, item="Discharge", reach="reach_2")
    """

    def __init__(
        self,
        data: PointType,
        *,
        reach: str,
        item: int | str | None = None,
        name: str | None = None,
        weight: float = 1.0,
        quantity: Quantity | None = None,
        aux_items: list[int | str] | None = None,
        attrs: dict | None = None,
    ) -> None:
        if not self._is_input_validated(data):
            data = _parse_network_breakpoint_input(
                data,
                name=name,
                item=item,
                quantity=quantity,
                aux_items=aux_items,
                reach=reach,
                distance=None,
            )
        assert isinstance(data, xr.Dataset)
        super().__init__(data=data, weight=weight, attrs=attrs)

    @property
    def reach(self) -> str:
        """Reach ID of this observation."""
        return str(self.data.coords["reach"].item())

    def _create_new_instance(self, data: xr.Dataset) -> Self:
        """Reconstruct instance from a dataset slice."""
        return self.__class__(data, reach=str(data.coords["reach"].item()))

    @overload
    @classmethod
    def from_multiple(
        cls,
        *,
        data: PointType,
        reaches: dict[str, str | int],
        quantity: Quantity | None = None,
        aux_items: list[int | str] | None = None,
        attrs: dict | None = None,
    ) -> list[ReachObservation]: ...

    @overload
    @classmethod
    def from_multiple(
        cls,
        *,
        reaches: dict[str, PointType],
        quantity: Quantity | None = None,
        aux_items: list[int | str] | None = None,
        attrs: dict | None = None,
    ) -> list[ReachObservation]:
        pass

    @overload
    @classmethod
    def from_multiple(
        cls,
        *,
        data: PointType,
        db: str | Path | Any,
        quantity: Quantity | str | None = None,
        source: str | None = None,
        on_missing: Literal["raise", "skip"] = "raise",
        aux_items: list[int | str] | None = None,
        attrs: dict | None = None,
    ) -> list[ReachObservation]: ...

    @classmethod
    def from_multiple(
        cls,
        *,
        data: PointType | None = None,
        reaches: dict[str, Any] | None = None,
        db: str | Path | Any | None = None,
        quantity: Quantity | str | None = None,
        source: str | None = None,
        on_missing: Literal["raise", "skip"] = "raise",
        aux_items: list[int | str] | None = None,
        attrs: dict | None = None,
    ) -> list[ReachObservation]:
        """Create multiple ReachObservation objects.

        Two calling conventions are supported:

        1. **Separate data sources** — pass only ``reaches`` as a dict mapping
           each reach ID to its own data source (file path, DataFrame, etc.)::

               obs = ReachObservation.from_multiple(reaches={"r1": df1, "r2": "sensor.csv"})

        2. **Shared data source** — pass a single ``data`` object together with
           ``reaches`` as a dict mapping each reach ID to the column name or
           index to select from ``data``::

               obs = ReachObservation.from_multiple(data=df, reaches={"r1": "col_a", "r2": "col_b"})

        3. **MIKE+ database** — pass a single ``data`` object together with
           ``db``, and the reaches are looked up in the database::

               obs = ReachObservation.from_multiple(data="calib.dfs0", db="model.sqlite")

           One observation is created per item of ``data`` that the database
           places on a link without a chainage.

        Parameters
        ----------
        data : PointType, optional
            Shared data source (required when ``reaches`` values are column
            selectors, and when ``db`` is given).
        reaches : dict[str, PointType | str | int]
            Mapping of reach_id -> data source or column selector.

            Note that a reach can appear only once, so this form cannot express
            several observations on the same reach. Use ``db`` when the data has
            several sensors on one reach.
        db : str, Path or sqlite3.Connection, optional
            MIKE+ database locating the items of ``data`` in the network.
            Mutually exclusive with ``reaches``.
        quantity : Quantity or str, optional
            Physical quantity metadata, by default None. With ``db``, a string
            selects which quantity to build observations for and the metadata
            comes from the database; omit it and the quantity is inferred when
            the data holds only one.
        source : str, optional
            With ``db``, the file the items come from. Taken from ``data`` when
            that is a path, by default None.
        on_missing : {"raise", "skip"}, optional
            With ``db``, what to do with items the database cannot place, by
            default "raise".
        aux_items : list[int | str] | None, optional
            Auxiliary items, by default None.
        attrs : dict | None, optional
            Additional attributes, by default None.

        Returns
        -------
        list[ReachObservation]
            List of ReachObservation objects.

        Raises
        ------
        ValueError
            If both ``reaches`` and ``db`` are given, if neither is, or if the
            database cannot resolve the requested items.
        """
        if db is not None:
            if reaches is not None:
                raise ValueError(
                    "'reaches' and 'db' are mutually exclusive: the database "
                    "supplies the locations."
                )
            if data is None:
                raise ValueError("'data' is required when 'db' is given")
            return _observations_from_mikeplus(
                cls,
                data=data,
                db=db,
                kind="reach",
                location_arg="reach",
                quantity=quantity,
                source=source,
                on_missing=on_missing,
                aux_items=aux_items,
                attrs=attrs,
            )

        if isinstance(quantity, str):
            raise TypeError(
                "'quantity' must be a Quantity unless 'db' is given, got str"
            )

        if reaches is None:
            raise ValueError("'reaches' argument is required")
        if not isinstance(reaches, dict):
            raise TypeError(
                f"'reaches' must be a dict mapping reach_id -> data_source, got {type(reaches).__name__}"
            )

        reach_ids = list(reaches.keys())

        if data is None:
            data_sources: list[PointType] = list(reaches.values())
            return [
                cls(
                    data_i,
                    reach=reach_i,
                    item=None,
                    quantity=quantity,
                    aux_items=aux_items,
                    attrs=attrs,
                )
                for data_i, reach_i in zip(data_sources, reach_ids)
            ]
        else:
            reach_items: list[int | str | None] = list(reaches.values())
            return [
                cls(
                    data,
                    reach=reach_i,
                    item=item_i,
                    quantity=quantity,
                    aux_items=aux_items,
                    attrs=attrs,
                )
                for reach_i, item_i in zip(reach_ids, reach_items)
            ]


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
    GeometryType.VERTICAL: VerticalObservation,
    GeometryType.NODE: NodeObservation,
    GeometryType.REACH: ReachObservation,
}
