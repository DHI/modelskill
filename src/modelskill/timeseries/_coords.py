import numpy as np


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
    def __init__(
        self,
        node: int | str | None = None,
        boundary: str | None = None,
    ):
        self.node = node if node is not None else np.nan
        self.boundary = boundary if boundary is not None else np.nan

    @property
    def as_dict(self) -> dict:
        return {"node": self.node, "boundary": self.boundary}


class BreakpointCoords:
    def __init__(
        self,
        edge: str,
        distance: float,
        boundary: str | None = None,
    ):
        self.edge = edge
        self.distance = distance
        self.boundary = boundary if boundary is not None else np.nan

    @property
    def as_dict(self) -> dict:
        return {"edge": self.edge, "distance": self.distance, "boundary": self.boundary}


class EdgeCoords:
    """Coordinate for an edge-level observation (no specific chainage/distance)."""

    def __init__(self, edge: str):
        self.edge = edge

    @property
    def as_dict(self) -> dict:
        return {"edge": self.edge}
