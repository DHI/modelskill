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
__version__ = "1.0.a1"

if "64" not in architecture()[0]:
    raise Exception("This library has not been tested for a 32 bit system.")

from .types import Quantity
from .model.factory import ModelResult
from .model import PointModelResult, TrackModelResult, GridModelResult, DfsuModelResult
from .observation import PointObservation, TrackObservation
from .connection import compare, Connector, from_matched
from .settings import options, get_option, set_option, reset_option, load_style
from .plot import plot_temporal_coverage, plot_spatial_overview

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
    "plot_temporal_coverage",
    "plot_spatial_overview",
    "from_config",
]


def from_config(
    configuration: Union[dict, str], *, validate_eum=True, relative_path=True
):
    return Connector.from_config(
        configuration, validate_eum=validate_eum, relative_path=relative_path
    )
