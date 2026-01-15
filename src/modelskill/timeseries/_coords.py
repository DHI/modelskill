import numpy as np
from typing import Optional, Literal


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
        reach: Optional[str] = None,
        chainage: Optional[str | float] = None,
        gridpoint: Optional[int | Literal["start", "end"]] = None,
    ):
        self.node = node if node is not None else np.nan
        self.reach = reach if reach is not None else np.nan
        self.chainage = chainage if chainage is not None else np.nan
        self.gridpoint = gridpoint if gridpoint is not None else np.nan

        # TODO: add validation (move parse_network_input preamble)

    @property
    def as_dict(self) -> dict:
        return {
            "node": self.node,
            "reach": self.reach,
            "chainage": self.chainage,
            "gridpoint": self.gridpoint,
        }
