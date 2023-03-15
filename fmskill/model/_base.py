from pathlib import Path
import warnings
from typing import Optional, Union

import pandas as pd

from fmskill import types, utils
from fmskill.comparison import PointComparer, SingleObsComparer, TrackComparer
from fmskill.observation import Observation, PointObservation, TrackObservation


class ModelResultBase:
    def __init__(
        self,
        data: types.DataInputType,
        name: str,
        item: Optional[str] = None,
        itemInfo=None,
        quantity: Optional[str] = None,
        **kwargs,
    ) -> None:

        self.data = data
        self.item = item
        if name is None:
            raise ValueError("name must be specified!")
        self.name = name
        self.quantity = quantity
        self.itemInfo = utils.parse_itemInfo(itemInfo)

    def __repr__(self):
        txt = [f"<{self.__class__.__name__}> '{self.name}'"]
        txt.append(f"- Item: {self.item}")  # TODO: only if item is not None?
        return "\n".join(txt)

    @property
    def item_name(self):
        # backwards compatibility, delete?
        return self.item

    @staticmethod
    def _default_name(data) -> str:
        if isinstance(data, (str, Path)):
            return Path(data).stem

    @property
    def time(self) -> pd.DatetimeIndex:
        if hasattr(self.data, "time"):
            return pd.DatetimeIndex(self.data.time)
        elif hasattr(self.data, "index"):
            return pd.DatetimeIndex(self.data.index)
        else:
            raise AttributeError("Could not extract time from data")

    @property
    def start_time(self) -> pd.Timestamp:
        if hasattr(self.data, "start_time"):
            return pd.Timestamp(self.data.start_time)
        else:
            return self.time[0]

    @property
    def end_time(self) -> pd.Timestamp:
        if hasattr(self.data, "end_time"):
            return pd.Timestamp(self.data.end_time)
        else:
            return self.time[-1]

    def _validate_start_end(self, observation: Observation) -> bool:
        if observation.end_time < self.start_time:
            return False
        if observation.start_time > self.end_time:
            return False
        return True

    def _validate_observation(self, observation: Observation):
        ok = utils.validate_item_eum(self.itemInfo, observation)
        if not ok:
            raise ValueError("Could not extract observation")

    # TODO: does not do anything except validation???
    def extract_observation(
        self, observation: Union[PointObservation, TrackObservation], validate=True
    ) -> SingleObsComparer:
        """Extract ModelResult at observation for comparison

        Parameters
        ----------
        observation : <PointObservation> or <TrackObservation>
            points and times at which modelresult should be extracted
        validate: bool, optional
            Validate if observation is inside domain and that eum type
            and units match; Default: True

        Returns
        -------
        <fmskill.SingleObsComparer>
            A comparer object for further analysis or plotting
        """

        if validate:
            self._validate_observation(observation)
