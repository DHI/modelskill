import warnings
from typing import Union

import pandas as pd

from fmskill import types, utils
from fmskill.comparison import PointComparer, SingleObsComparer, TrackComparer
from fmskill.observation import PointObservation, TrackObservation


class ModelResultBase:
    def __init__(
        self,
        data: Union[types.ExtractableType, pd.DataFrame],
        item: str = None,
        itemInfo=None,
        name: str = None,
        quantity: str = None,
    ) -> None:

        self.data = data
        self.item = item
        self.name = name
        self.quantity = quantity
        self.itemInfo = utils.parse_itemInfo(itemInfo)

    def __repr__(self):
        txt = [f"<{self.__class__.__name__}> '{self.name}'"]
        txt.append(f"- Item: {self.item}")
        return "\n".join(txt)

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
            # ok = self._validate_observation(observation)
            # if ok:
            ok = utils._validate_item_eum(self.itemInfo, observation)
            if not ok:
                raise ValueError("Could not extract observation")

        if isinstance(observation, PointObservation):
            point_mr = self.extract(observation)
            comparer = PointComparer(observation, point_mr.data)
        elif isinstance(observation, TrackObservation):
            track_mr = self.extract(observation)
            comparer = TrackComparer(observation, track_mr.data)
        else:
            raise ValueError("Only point and track observation are supported!")

        if len(comparer.data) == 0:
            warnings.warn(f"No overlapping data in found for obs '{observation.name}'!")
            comparer = None

        return comparer
