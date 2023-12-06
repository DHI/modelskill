from __future__ import annotations
from typing import List, Optional, Protocol, Sequence
from dataclasses import dataclass
import warnings

import pandas as pd

from ..utils import _get_name
from ..observation import Observation, PointObservation, TrackObservation


@dataclass
class SelectedItems:
    values: str
    aux: list[str]

    @property
    def all(self) -> List[str]:
        return [self.values] + self.aux


def _parse_items(
    avail_items: Sequence[str],
    item: int | str | None,
    aux_items: Optional[Sequence[int | str]] = None,
) -> SelectedItems:
    """If input has exactly 1 item we accept item=None"""
    if item is None:
        if len(avail_items) == 1:
            item = 0
        elif len(avail_items) > 1:
            raise ValueError(
                f"Input has more than 1 item, but item was not given! Available items: {avail_items}"
            )

    item = _get_name(item, valid_names=avail_items)
    if isinstance(aux_items, (str, int)):
        aux_items = [aux_items]
    aux_items_str = [_get_name(i, valid_names=avail_items) for i in aux_items or []]

    # check that there are no duplicates
    res = SelectedItems(values=item, aux=aux_items_str)
    if len(set(res.all)) != len(res.all):
        raise ValueError(f"Duplicate items! {res.all}")

    return res


def _validate_overlap_in_time(time: pd.DatetimeIndex, observation: Observation) -> None:
    overlap_in_time = (
        time[0] <= observation.time[-1] and time[-1] >= observation.time[0]
    )
    if not overlap_in_time:
        warnings.warn(
            f"No time overlap. Observation '{observation.name}' outside model time range! "
        )


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
