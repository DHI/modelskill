import warnings
import mikeio
import numpy as np
import pandas as pd
import xarray as xr

from fmskill import utils
from fmskill.model import PointModelResult, TrackModelResult, protocols
from fmskill.observation import PointObservation, TrackObservation


def _xy_in_xr_domain(data: xr.Dataset, x: float, y: float) -> bool:
    if (x is None) or (y is None):
        raise ValueError(f"Cannot determine if point ({x}, {y}) is inside domain!")
    xmin = data.x.values.min()
    xmax = data.x.values.max()
    ymin = data.y.values.min()
    ymax = data.y.values.max()
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
