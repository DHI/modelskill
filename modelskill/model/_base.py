from __future__ import annotations
from collections import Counter
from typing import List, Optional, Protocol, Sequence, TYPE_CHECKING
from dataclasses import dataclass
import warnings

import pandas as pd

if TYPE_CHECKING:
    from .point import PointModelResult
    from .track import TrackModelResult

from ..utils import _get_name
from ..obs import Observation, PointObservation, TrackObservation


@dataclass
class SelectedItems:
    values: str
    aux: list[str]

    @property
    def all(self) -> List[str]:
        return [self.values] + self.aux

    @staticmethod
    def parse(
        avail_items: Sequence[str],
        item: int | str | None,
        aux_items: Optional[Sequence[int | str]] = None,
    ) -> SelectedItems:
        return _parse_items(avail_items, item, aux_items)


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
        element_counts = Counter(res.all)
        duplicates = [element for element, count in element_counts.items() if count > 1]
        raise ValueError(f"Duplicate items! {duplicates}")

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
    def extract(
        self,
        observation: PointObservation | TrackObservation,
        spatial_method: Optional[str] = None,
    ) -> PointModelResult | TrackModelResult:
        ...

    def _extract_point(
        self, observation: PointObservation, spatial_method: Optional[str] = None
    ) -> PointModelResult:
        ...

    def _extract_track(
        self, observation: TrackObservation, spatial_method: Optional[str] = None
    ) -> TrackModelResult:
        ...
