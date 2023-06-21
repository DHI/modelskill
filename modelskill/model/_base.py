import warnings
from typing import Optional

import pandas as pd

from ..observation import Observation
from ..types import Quantity, DataInputType


class ModelResultBase:
    def __init__(
        self,
        data: DataInputType,
        name: str,
        quantity: Optional[Quantity] = None,
    ) -> None:

        self.data = data
        self.name = name
        self.quantity = Quantity.undefined() if quantity is None else quantity

    def __repr__(self):
        return f"<{self.__class__.__name__}> '{self.name}'"

    @staticmethod
    def _any_obs_in_model_time(
        time_obs: pd.DatetimeIndex, time_model: pd.DatetimeIndex
    ) -> bool:
        """Check if any observation times are in model time range"""
        return (time_obs[-1] >= time_model[0]) & (time_obs[0] <= time_model[-1])

    def _validate_any_obs_in_model_time(
        self, obs_name: str, time_obs: pd.DatetimeIndex, time_model: pd.DatetimeIndex
    ) -> None:
        """Check if any observation times are in model time range"""
        ok = self._any_obs_in_model_time(time_obs, time_model)
        if not ok:
            # raise ValueError(
            warnings.warn(
                f"No time overlap. Observation '{obs_name}' outside model time range! "
                + f"({time_obs[0]} - {time_obs[-1]}) not in ({time_model[0]} - {time_model[-1]})"
            )

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
