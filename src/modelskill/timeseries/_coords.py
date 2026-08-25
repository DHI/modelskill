from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

#: Scalar coordinates that say where a network timeseries sits, rather than what
#: it holds. They are dropped on the way to a dataframe, where they would
#: otherwise become columns.
NETWORK_LOCATION_COORDS = ("node", "node_index", "reach", "distance")


def location_from_coords(ds: xr.Dataset) -> Any:
    """Where a network timeseries sits, as the network that produced it named it.

    Returns a node name for a node, a ``(reach, distance)`` pair for a
    breakpoint, a reach name when no distance was given, and None for data that
    carries no network location. The value is returned as recorded, so a comparer
    saved by an older version gives back the integer it stored.
    """
    if "node" in ds.coords:
        return _scalar(ds, "node")
    if "reach" in ds.coords:
        reach = _scalar(ds, "reach")
        if "distance" not in ds.coords:
            return reach
        return (reach, _scalar(ds, "distance"))
    return None


def _scalar(ds: xr.Dataset, name: str) -> Any:
    value = np.atleast_1d(ds.coords[name].values)[0]
    return value.item() if hasattr(value, "item") else value


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
    def __init__(self, node: int | str | None = None):
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
