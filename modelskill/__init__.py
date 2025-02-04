from pathlib import Path
from platform import architecture
from typing import Union


# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = "1.2.dev0"

if "64" not in architecture()[0]:
    raise Exception("This library has not been tested for a 32 bit system.")

from .quantity import Quantity
from .model import model_result
from .model import (
    PointModelResult,
    TrackModelResult,
    GridModelResult,
    DfsuModelResult,
    DummyModelResult,
)
from .obs import observation, PointObservation, TrackObservation
from .matching import from_matched, match
from .configuration import from_config
from .settings import options, get_option, set_option, reset_option, load_style
from . import plotting
from . import data
from .comparison import ComparerCollection, Comparer
from .skill import SkillTable
from .timeseries import TimeSeries


def load(filename: Union[str, Path]) -> ComparerCollection:
    """Load a ComparerCollection from a netcdf file.

    Parameters
    ----------
    filename : str or Path
        Filename of the nc file.

    Returns
    -------
    ComparerCollection
        The loaded ComparerCollection.

    Examples
    --------
    >>> cc = ms.match(obs, mod)
    >>> cc.save("my_comparer_collection.nc")
    >>> cc2 = ms.load("my_comparer_collection.nc")"""

    return ComparerCollection.load(filename)


__all__ = [
    "Quantity",
    "model_result",
    "PointModelResult",
    "TrackModelResult",
    "GridModelResult",
    "DfsuModelResult",
    "DummyModelResult",
    "observation",
    "PointObservation",
    "TrackObservation",
    "TimeSeries",
    "match",
    "from_matched",
    "Comparer",
    "ComparerCollection",
    "SkillTable",
    "options",
    "get_option",
    "set_option",
    "reset_option",
    "load_style",
    "plotting",
    "from_config",
    "data",
]
