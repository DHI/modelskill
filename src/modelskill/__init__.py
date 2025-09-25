from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from platform import architecture
from typing import Union
import zipfile


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
try:
    # read version from installed package
    __version__ = version("modelskill")
except PackageNotFoundError:
    # package is not installed
    __version__ = "dev"

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


def load(filename: Union[str, Path]) -> Comparer | ComparerCollection:
    """Load a Comparer or ComparerCollection from a netcdf/zip file.

    Parameters
    ----------
    filename : str or Path
        Filename of the netcdf or zip file to load.

    Returns
    -------
    Comparer or ComparerCollection
        The loaded Comparer or ComparerCollection.


    Examples
    --------
    >>> cc = ms.match(obs, mod)
    >>> cc.save("my_comparer_collection.msk")
    >>> cc2 = ms.load("my_comparer_collection.msk")"""

    try:
        return ComparerCollection.load(filename)
    except zipfile.BadZipFile:
        try:
            return Comparer.load(filename)
        except Exception as e:
            raise ValueError(
                f"File '{filename}' is neither a valid zip archive nor a NetCDF file: {e}"
            )


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
    "load",
]
