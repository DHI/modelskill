from platform import architecture
from typing import Union
import yaml


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
__version__ = "0.1.2.dev"

if "64" not in architecture()[0]:
    raise Exception("This library has not been tested for a 32 bit system.")

from .model import ModelResult, ModelResultCollection
from .observation import PointObservation, TrackObservation


def create(configuration: Union[dict, str]):

    if isinstance(configuration, str):
        with open(configuration) as f:
            contents = f.read()
        configuration = yaml.load(contents, Loader=yaml.FullLoader)

    mr = ModelResult(filename=configuration["filename"], name=configuration.get("name"))
    for connection in configuration["observations"]:
        observation = connection["observation"]

        if observation.get("type") == "track":
            obs = TrackObservation(**observation)
        else:
            obs = PointObservation(**observation)

        mr.add_observation(obs, item=connection["item"])

    return mr
