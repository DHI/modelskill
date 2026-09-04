"""Resolve dfs0 items to network locations using a MIKE+ database.

A MIKE+ project ships a sqlite database alongside its result files. Two of its
tables describe where the measured timeseries belong in the network:

* ``m_Measurement`` — one row per measured timeseries, naming the file
  (``tsfilename``) and the item within it (``tsitemname``), plus the modelled
  quantity (``resitemname``).
* ``m_Station`` — the location, as ``locationid`` plus a ``locationtype``
  saying whether that identifier names a node or a link.

:func:`resolve_stations` joins the two and returns one row per timeseries item.
Everything MIKE+-specific — table names, the join, the ``locationtype`` codes,
the encoding of ``resitemname`` — is contained in this module. Callers see only
the columns listed in :data:`CONTRACT_COLUMNS`, so a change to the database
layout is a change to this file alone.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

import pandas as pd

Kind = Literal["node", "reach"]

#: Columns returned by :func:`resolve_stations`, one row per timeseries item.
#:
#: ``item_name``  name of the item in the data source
#: ``name``       display name for the observation
#: ``location``   node alias, or ``(reach_id, distance)`` for a breakpoint
#: ``kind``       ``"node"`` or ``"reach"``, i.e. which observation class fits
#: ``quantity``   modelled quantity name
CONTRACT_COLUMNS = ["item_name", "name", "location", "kind", "quantity"]

# MIKE+ m_Station.locationtype codes. 8 is a junction and 12 a tank or
# reservoir; both are graph nodes. 9 is a link, which becomes a breakpoint when
# the station carries a chainage and a whole reach when it does not. Unknown
# codes raise rather than guess.
_FAMILY_BY_LOCATIONTYPE: dict[int, str] = {8: "node", 9: "link", 12: "node"}

_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "m_Station": {"muid", "locationid", "locationtype", "chainagevalue", "assetname"},
    "m_Measurement": {
        "measurementstationid",
        "tsfilename",
        "tsitemname",
        "resitemname",
    },
}

_JOIN_QUERY = """
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


@contextmanager
def _connect(db: str | Path | sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    if isinstance(db, sqlite3.Connection):
        yield db
    else:
        conn = sqlite3.connect(str(db))
        try:
            yield conn
        finally:
            conn.close()


def _validate_schema(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if missing := sorted(set(_REQUIRED_COLUMNS) - tables):
        raise ValueError(
            f"Database is missing table(s) {missing}. "
            "A MIKE+ database with 'm_Station' and 'm_Measurement' is required."
        )
    for table, required in _REQUIRED_COLUMNS.items():
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info([{table}])")}
        if missing_cols := sorted(required - columns):
            raise ValueError(
                f"Table '{table}' is missing column(s) {missing_cols}. "
                "The database layout is not the one modelskill expects."
            )


def _read_join(conn: sqlite3.Connection, source: str | None) -> pd.DataFrame:
    query = _JOIN_QUERY
    params: list[str] = []
    if source is not None:
        query += " WHERE m.tsfilename LIKE ?"
        params.append(f"%{Path(source).name}%")
    return pd.read_sql_query(query, conn, params=params)


def _kind_and_location(row: pd.Series) -> tuple[Kind, str | tuple[str, float]]:
    # A link station with a chainage names a point along a reach, which is a
    # node observation at a breakpoint. Without a chainage it names the reach
    # as a whole.
    location_type = row["locationtype"]
    try:
        family = _FAMILY_BY_LOCATIONTYPE[int(location_type)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"Station '{row['locationid']}' has unsupported locationtype "
            f"{location_type!r}. Known codes are {sorted(_FAMILY_BY_LOCATIONTYPE)}."
        ) from None

    chainage = row["chainagevalue"]
    if family == "node":
        return "node", str(row["locationid"])
    if pd.isna(chainage):
        return "reach", str(row["locationid"])
    return "node", (str(row["locationid"]), float(chainage))


def _describe_unresolved(
    conn: sqlite3.Connection, missing: Sequence[str], source: str | None
) -> str:
    assets = set(
        pd.read_sql_query("SELECT assetname FROM m_Station", conn)["assetname"]
        .dropna()
        .tolist()
    )
    known = [item for item in missing if item in assets]
    unknown = [item for item in missing if item not in assets]

    lines = []
    if known:
        where = f" for '{Path(source).name}'" if source else ""
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


def _choose_names(selection: pd.DataFrame) -> pd.Series:
    # assetname is far shorter than the raw item name and is normally unique,
    # but it is only safe as a display name when it distinguishes every row.
    assets = selection["assetname"]
    if assets.notna().all() and assets.nunique() == len(selection):
        return assets.astype(str)
    return selection["item_name"].astype(str)


def resolve_stations(
    db: str | Path | sqlite3.Connection,
    *,
    item_names: Iterable[str],
    source: str | None = None,
    quantity: str | None = None,
    kind: Kind | None = None,
    on_missing: Literal["raise", "skip"] = "raise",
) -> pd.DataFrame:
    """Resolve data source items to network locations via a MIKE+ database.

    Parameters
    ----------
    db : str, Path or sqlite3.Connection
        MIKE+ database, as a path or an already-open connection.
    item_names : Iterable[str]
        Item names to resolve, e.g. the column names of a dfs0.
    source : str, optional
        File the items come from, matched against ``tsfilename``. Only the file
        name is used, so a full path is fine. By default None, in which case
        measurements from every file are considered and an item registered
        against more than one file raises.
    quantity : str, optional
        Quantity to select, e.g. ``"Pressure"``. By default None, in which case
        the quantity is inferred when the selection holds only one and raises
        when it holds several.
    kind : {"node", "reach"}, optional
        Restrict to locations of this kind, by default None (no restriction).
    on_missing : {"raise", "skip"}, optional
        What to do with items that resolve to no measurement at all, by default
        "raise".

    Returns
    -------
    pd.DataFrame
        One row per resolved item, with the columns in
        :data:`CONTRACT_COLUMNS`.

    Raises
    ------
    ValueError
        If the database layout is not the expected one, if items cannot be
        resolved and ``on_missing="raise"``, if the quantity is ambiguous or
        absent, or if no location of the requested kind remains.
    """
    requested = list(dict.fromkeys(item_names))

    with _connect(db) as conn:
        _validate_schema(conn)
        rows = _read_join(conn, source)
        rows["quantity"] = rows["resitemname"].str.split(";").str[0].str.strip()
        rows = rows[rows["item_name"].isin(requested)].copy()

        missing = [item for item in requested if item not in set(rows["item_name"])]
        if missing and on_missing == "raise":
            raise ValueError(
                f"{len(missing)} of {len(requested)} items could not be resolved "
                f"against the MIKE+ database.\n"
                + _describe_unresolved(conn, missing, source)
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

    resolved = rows.apply(_kind_and_location, axis=1)
    rows["kind"] = [k for k, _ in resolved]
    rows["location"] = [location for _, location in resolved]

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
                f"{pool['quantity'].value_counts().to_dict()}. Pass one of {available}."
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

    selection = selection.copy()
    selection["name"] = _choose_names(selection)
    return selection[CONTRACT_COLUMNS].reset_index(drop=True)
