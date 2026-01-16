import warnings

import numpy as np
import pandas as pd

from mikeio1d import Res1D
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


def read_network_coords(
    data: Res1D, coords: NetworkCoords, variable: Optional[str] = None
) -> pd.DataFrame:
    def variable_name_to_res1d(name: str) -> str:
        return name.replace(" ", "").replace("_", "")

    if ("reaches" not in dir(data)) or ("nodes" not in dir(data)):
        raise ValueError(
            "Invalid file format. Data must have a network structure containing 'nodes' and 'reaches'."
        )

    if coords.by_node and not coords.by_reach:
        location = data.nodes[str(coords.node)]

    if coords.by_reach and not coords.by_node:
        location = data.reaches[coords.reach][coords.chainage]

    df = location.to_dataframe()
    if variable is None:
        if len(df.columns) != 1:
            raise ValueError(
                f"The network location does not have a unique quantity: {location.columns}, in such case 'variable' argument cannot be None"
            )
        return df
    else:
        res1d_name = variable_name_to_res1d(variable)
        relevant_columns = [col for col in df.columns if res1d_name in col]
        assert len(relevant_columns) == 1
        return df.rename(columns={relevant_columns[0]: res1d_name})
