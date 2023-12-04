from typing import Protocol

from ..observation import Observation, PointObservation, TrackObservation


class SpatialField(Protocol):
    """Protocol for 2d spatial fields (grids and unstructured meshes)

    Methods
    -------
    extract(observation: Observation)
    extract_point(observation: PointObservation)
    extract_track(observation: TrackObservation)
    """

    def extract(self, observation: Observation):
        ...

    def extract_point(self, observation: PointObservation):
        ...

    def extract_track(self, observation: TrackObservation):
        ...
