import os
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
from collections.abc import Mapping

from ..comparison import BaseComparer


class ModelResultInterface(ABC):  # pragma: no cover
    @property
    @abstractmethod
    def start_time(self):
        pass

    @property
    @abstractmethod
    def end_time(self):
        pass

    @property
    @abstractmethod
    def item_name(self):
        pass

    # @property
    # def itemInfo(self):
    #     return None

    @abstractmethod
    def extract_observation(self, observation) -> BaseComparer:
        pass


class MultiItemModelResult(ABC, Mapping):
    _mr_items = None

    @property
    @abstractmethod
    def item_names(self):
        pass

    @property
    @abstractmethod
    def _get_item_name(self):
        pass

    def __getitem__(self, x):
        if isinstance(x, (int, str)):
            x = self._get_item_name(x)
        return self._mr_items[x]

    def __len__(self) -> int:
        return len(self._mr_items)

    def __iter__(self):
        return iter(self._mr_items.values())

    @property
    @abstractmethod
    def start_time(self):
        pass

    @property
    @abstractmethod
    def end_time(self):
        pass
