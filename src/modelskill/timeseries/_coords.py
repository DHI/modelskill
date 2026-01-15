import warnings

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
        self.node = node
        self.reach = reach
        self.chainage = chainage
        self.gridpoint = gridpoint
        self.validate_coordinates()

    @property
    def as_dict(self) -> dict:
        return {
            "node": self.node,
            "reach": self.reach,
            "chainage": self.chainage,
            "gridpoint": self.gridpoint,
        }

    @property
    def by_node(self) -> bool:
        return self.node is not None

    @property
    def by_reach(self) -> bool:
        return self.reach is not None

    @property
    def with_chainage(self) -> bool:
        return self.chainage is not None

    @property
    def with_index(self) -> bool:
        return self.gridpoint is not None

    def validate_coordinates(self):
        if self.by_node and not self.by_reach:
            if self.with_chainage or self.with_index:
                warnings.warn(
                    "'chainage' or 'gridpoint' are only relevant when passed with 'reach' but they were passed with 'node', so they will be ignored."
                )

        elif self.by_reach and not self.by_node:
            if self.with_index == self.with_chainage:
                raise ValueError(
                    "Locations accessed by chainage must be specified either by chainage or by index, not both."
                )

        else:
            raise ValueError(
                "A network location must be specified either by node or by reach."
            )
