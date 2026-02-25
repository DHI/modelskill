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


class NetworkCoords:
    def __init__(
        self,
        node: int | None = None,
        boundary: str | None = None,
    ):
        self.node = node if node is not None else np.nan
        self.boundary = boundary if boundary is not None else np.nan

    @property
    def as_dict(self) -> dict:
        return {"node": self.node, "boundary": self.boundary}
