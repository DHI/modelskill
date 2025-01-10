"""
# Model Result

A model result can either be a simple point/track, or spatial field (e.g. 2d dfsu file) from which data can be *extracted* at the observation positions by spatial interpolation. The following types are available:

* Timeseries
    - [`PointModelResult`](`modelskill.PointModelResult`) - a point result from a dfs0/nc file or a DataFrame
    - [`TrackModelResult`](`modelskill.TrackModelResult`) - a track (moving point) result from a dfs0/nc file or a DataFrame
* SpatialField (extractable)
    - [`GridModelResult`](`modelskill.GridModelResult`) - a spatial field from a dfs2/nc file or a Xarray Dataset
    - [`DfsuModelResult`](`modelskill.DfsuModelResult`) - a spatial field from a dfsu file

A model result can be created by explicitly invoking one of the above classes or using the [`model_result()`](`modelskill.model_result`) function which will return the appropriate type based on the input data (if possible).
"""

# from .factory import ModelResult

from .factory import model_result
from .point import PointModelResult
from .track import TrackModelResult
from .dfsu import DfsuModelResult
from .grid import GridModelResult
from .dummy import DummyModelResult

__all__ = [
    "PointModelResult",
    "TrackModelResult",
    "DfsuModelResult",
    "GridModelResult",
    "model_result",
    "DummyModelResult",
]
