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
__version__ = "1.0.a2"

if "64" not in architecture()[0]:
    raise Exception("This library has not been tested for a 32 bit system.")

from .types import Quantity
from .model.factory import ModelResult
from .model import PointModelResult, TrackModelResult, GridModelResult, DfsuModelResult
from .observation import PointObservation, TrackObservation
from .matching import compare, from_matched
from .connection import Connector
from .settings import options, get_option, set_option, reset_option, load_style
from . import plotting
from .comparison import ComparerCollection


def load(filename: Union[str, Path]) -> ComparerCollection:
    """Load a ComparerCollection from a zip file.

    Parameters
    ----------
    filename : str or Path
        Filename of the zip file.

    Returns
    -------
    ComparerCollection
        The loaded ComparerCollection.

    Examples
    --------
    >>> cc = ms.compare(obs, mod)
    >>> cc.save("my_comparer_collection.msk")
    >>> cc2 = ms.load("my_comparer_collection.msk")"""

    return ComparerCollection.load(filename)


__all__ = [
    "Quantity",
    "ModelResult",
    "PointModelResult",
    "TrackModelResult",
    "GridModelResult",
    "DfsuModelResult",
    "PointObservation",
    "TrackObservation",
    "compare",
    "Connector",
    "from_matched",
    "options",
    "get_option",
    "set_option",
    "reset_option",
    "load_style",
    "plotting",
    "from_config",
]


def from_config(
    configuration: Union[dict, str], *, validate_eum=True, relative_path=True
):
    return Connector.from_config(
        configuration, validate_eum=validate_eum, relative_path=relative_path
    )
