from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from modelskill.model.point import PointModelResult
from modelskill.model.track import TrackModelResult
from modelskill.obs import PointObservation, TrackObservation


@dataclass
class DummyModelResult:
    name: str = "dummy"
    data: float | None = None
    strategy: Literal["mean", "constant"] = "constant"
    """Dummy model result that always returns the same value.

    Similar in spirit to <https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html>

    Parameters
    ----------
    data : float, optional
        The value to return if strategy is 'constant', by default None
    name : str, optional
        The name of the model result, by default 'dummy'
    strategy : str, optional
        The strategy to use, 'mean' uses the mean of the observation, 'constant' uses the value given in data, by default 'constant'

    Examples
    --------
    >>> import pandas as pd
    >>> import modelskill as ms
    >>> obs = ms.PointObservation(pd.DataFrame([0.0, 1.0], index=pd.date_range("2000", freq="H", periods=2)), name="foo")
    >>> mr = ms.DummyModelResult(strategy='mean')
    >>> pmr = mr.extract(obs)
    >>> pmr.to_dataframe()
                        dummy
    time
    2000-01-01 00:00:00    0.5
    2000-01-01 01:00:00    0.5
    """

    def __post_init__(self):
        if self.strategy == "constant" and self.data is None:
            raise ValueError("data must be given when strategy is 'constant'")

    def extract(
        self,
        observation: PointObservation | TrackObservation,
        spatial_method: Optional[str] = None,
    ) -> PointModelResult | TrackModelResult:
        if spatial_method is not None:
            raise NotImplementedError(
                "spatial interpolation not possible when matching point model results with point observations"
            )

        da = observation.data[observation.name].copy()
        if self.strategy == "mean":
            da[:] = da.mean()
        else:
            da[:] = self.data

        if isinstance(observation, PointObservation):
            return PointModelResult(
                data=da, x=observation.x, y=observation.y, name=self.name
            )

        elif isinstance(observation, TrackObservation):
            data = pd.DataFrame(
                {
                    "x": observation.x,
                    "y": observation.y,
                    "value": da.values,
                },
                index=da.time,
            )
            return TrackModelResult(data=data, name=self.name)
        else:
            raise ValueError(
                f"observation must be a PointObservation or TrackObservation not {type(observation)}"
            )
