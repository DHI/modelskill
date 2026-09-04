from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

#: Scalar coordinates that say where a network timeseries sits, rather than what
#: it holds. They are dropped on the way to a dataframe, where they would
#: otherwise become columns.
NETWORK_LOCATION_COORDS = ("node", "node_index", "reach", "distance")


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
