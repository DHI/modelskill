from __future__ import annotations

import math
from typing import Any

import numpy as np
import xarray as xr


class XYZCoords:
    def __init__(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ):
        self.x = x if x is not None else np.nan
        self.y = y if y is not None else np.nan
        self.z = z

    @property
    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}


class NodeCoords:
    def __init__(self, node: str | None = None):
        self.node = node if node is not None else np.nan

    @property
    def as_dict(self) -> dict:
        return {"node": self.node}


class ReachCoords:
    """Coordinates for an observation along a network reach.

    Parameters
    ----------
    reach : str
        Reach identifier.
    distance : float or None, optional
        Along-reach distance (chainage).  When ``None`` the observation is
        reach-level (no specific chainage) and no ``distance`` coordinate is
        stored in the dataset.
    """

    def __init__(self, reach: str, distance: float | None = None):
        self.reach = reach
        self.distance = distance

    @property
    def as_dict(self) -> dict:
        d: dict = {"reach": self.reach}
        if self.distance is not None:
            d["distance"] = self.distance
        return d


def _coordinate_values(ds: xr.Dataset, coord: str) -> Any:
    """A dataset's values for one coordinate, or None when it has no such coordinate.

    A scalar coordinate is unwrapped to its single value; anything else is
    handed back as the array it is.
    """
    if coord not in ds.coords:
        return None
    vals = ds[coord].values
    return np.atleast_1d(vals)[0] if vals.ndim == 0 else vals


#: Scalar coordinates that say where a network timeseries sits, rather than what
#: it holds. They are dropped on the way to a dataframe, where they would
#: otherwise become columns.
NETWORK_LOCATION_COORDS = ("node", "node_index", "reach", "distance")


def network_location(ds: xr.Dataset) -> Any:
    """Where a network timeseries sits, as the network that produced it named it.

    Returns a node name for a node, a ``(reach, distance)`` pair for a
    breakpoint, a reach name when no distance was given, and None for data that
    carries no network location. The value is returned as recorded, so a comparer
    saved by an older version gives back the integer it stored.
    """
    if "node" in ds.coords:
        return _network_scalar(ds, "node")
    if "reach" in ds.coords:
        reach = _network_scalar(ds, "reach")
        if "distance" not in ds.coords:
            return reach
        return (reach, _network_scalar(ds, "distance"))
    return None


def _network_scalar(ds: xr.Dataset, name: str) -> Any:
    value = _coordinate_values(ds, name)
    return value.item() if hasattr(value, "item") else value


def _reject_conflicting_location(ds: xr.Dataset, named: Any, *, argument: str) -> None:
    """Raise when data that already knows where it sits is given another location.

    A dataset that has been through modelskill carries its own location
    coordinates, and the constructors cannot re-apply them, so a location named
    alongside it would be dropped without a word.

    The comparison is a plain one rather than a lookup: an observation has no
    network when it is built, so it cannot snap a near-miss distance the way
    :meth:`~modelskill.model.network.NetworkModelResult.extract` does.

    Parameters
    ----------
    ds : xr.Dataset
        Data that has already been through modelskill, and so carries its own
        location coordinates.
    named : str or tuple of (str, float)
        The location the caller named: a node name, a reach name, or a
        ``(reach, distance)`` break point.
    argument : str
        Name of the keyword the location came from, for the error message.

    Raises
    ------
    ValueError
        If the data carries no network location, or carries a different one.
    """
    carried = network_location(ds)
    if carried is None:
        raise ValueError(
            f"The data has been through modelskill but carries no network "
            f"location, so {argument!r} ({named!r}) has nothing to agree with. "
            "Build the observation from a DataFrame instead."
        )
    if not _same_location(carried, named):
        raise ValueError(
            f"The data already sits at {carried!r}, but {argument!r} says "
            f"{named!r}. A dataset that has been through modelskill carries its "
            f"own location: pass {argument}={carried!r}, or build the observation "
            "from a DataFrame to place it somewhere else."
        )


def _same_location(carried: Any, named: Any) -> bool:
    """Whether two network locations name the same place.

    Break point distances are compared closely rather than exactly, so that a
    location read off a dataset and handed straight back still agrees with
    itself.
    """
    if isinstance(carried, tuple) != isinstance(named, tuple):
        return False
    if isinstance(carried, tuple):
        return str(carried[0]) == str(named[0]) and math.isclose(
            float(carried[1]), float(named[1])
        )
    return str(carried) == str(named)
