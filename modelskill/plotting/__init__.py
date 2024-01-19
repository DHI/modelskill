"""The plotting module provides functions useful for skill assessment that can be used independently of the comparison module.

* `scatter` is a function that can be used to plot a scatter suitable for skill assessment, with a 1:1 line and a linear regression line.
* `wind_rose` is a function that can be used to plot a dual wind rose to compare two datasets of magnitudes and directions.
* `spatial_overview` is a function that can be used to plot a spatial overview of two datasets.
* `temporal_coverage` is a function that can be used to plot the temporal coverage of two datasets.
"""

from . import _settings  # noqa: F401
from ._scatter import scatter
from ._spatial_overview import spatial_overview
from ._taylor_diagram import taylor_diagram, TaylorPoint  # noqa: F401
from ._temporal_coverage import temporal_coverage
from ._wind_rose import wind_rose

__all__ = [
    "scatter",
    "spatial_overview",
    "taylor_diagram",
    "temporal_coverage",
    "wind_rose",
    "TaylorPoint",
]
