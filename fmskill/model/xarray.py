import pandas as pd
import xarray as xr
import warnings

from mikeio import eum
from ..observation import PointObservation
from ..comparison import PointComparer
from .abstract import ModelResultInterface, MultiItemModelResult


class _XarrayBase:
    ds = None
    itemInfo = eum.ItemInfo(eum.EUMType.Undefined)

    @property
    def start_time(self):
        return pd.Timestamp(self.ds.time.values[0])

    @property
    def end_time(self):
        return pd.Timestamp(self.ds.time.values[-1])

    def _get_item_name(self, item, item_names=None) -> str:
        raise NotImplementedError()

    def _get_item_num(self, item) -> int:
        raise NotImplementedError()


class XArrayModelResultItem(_XarrayBase, ModelResultInterface):
    @property
    def item_name(self):
        raise NotImplementedError()

    def __init__(self, ds, name: str = None, item=None):
        raise NotImplementedError()

    def __repr__(self):
        txt = [f"<XarrayModelResultItem> '{self.name}'"]
        txt.append(f"- Item: {self.item_name}")
        return "\n".join(txt)

    def extract_observation(self, observation: PointObservation) -> PointComparer:
        """Compare this ModelResult with an observation

        Parameters
        ----------
        observation : <PointObservation>
            Observation to be compared

        Returns
        -------
        <fmskill.PointComparer>
            A comparer object for further analysis or plotting
        """
        raise NotImplementedError()
        if isinstance(observation, PointObservation):
            pass
        else:
            raise ValueError("Only point observation are supported!")

        if len(comparer.df) == 0:
            warnings.warn(f"No overlapping data in found for obs '{observation.name}'!")
            comparer = None

        return comparer


class XArrayModelResult(_XarrayBase, MultiItemModelResult):
    @property
    def item_names(self):
        """List of item names (=data vars)"""
        return list(self.ds.data_vars)

    def __init__(self, input, name: str = None, item=None):
        raise NotImplementedError()
        # TODO: make sure it has a time coordinate
        # TODO: rename lat, lon, to x, y?
        self._mr_items = {}
        for it in self.item_names:
            self._mr_items[it] = XarrayModelResultItem(self.df, self.name, it)

    def __repr__(self):
        txt = [f"<XarrayModelResult> '{self.name}'"]
        for j, item in enumerate(self.item_names):
            txt.append(f"- Item: {j}: {item}")
        return "\n".join(txt)
