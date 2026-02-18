import numpy as np

from typing import Optional


class XYZCoords:
    def __init__(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
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
        node: Optional[int] = None,
        boundary: Optional[str] = None,
    ):
        self.node = node if node is not None else np.nan
        self.boundary = boundary if boundary is not None else np.nan

    @property
    def as_dict(self) -> dict:
        return {"node": self.node, "boundary": self.boundary}
