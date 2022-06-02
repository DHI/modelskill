from abc import ABC, abstractmethod
from collections.abc import Mapping
import pandas as pd
import mikeio

from ..comparison import BaseComparer


def _parse_itemInfo(itemInfo):
    if itemInfo is None:
        return mikeio.ItemInfo(mikeio.EUMType.Undefined)
    if isinstance(itemInfo, mikeio.ItemInfo):
        return itemInfo
    return mikeio.ItemInfo(itemInfo)


class ModelResultInterface(ABC):  # pragma: no cover
    @property
    @abstractmethod
    def start_time(self) -> pd.Timestamp:
        pass

    @property
    @abstractmethod
    def end_time(self) -> pd.Timestamp:
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

    def __repr__(self):
        txt = [f"<{self.__class__.__name__}> '{self.name}'"]
        txt.append(f"- Item: {self.item_name}")
        return "\n".join(txt)

    def _in_domain(self, x, y) -> bool:
        return True


class MultiItemModelResult(ABC, Mapping):  # pragma: no cover
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

    def __repr__(self):
        txt = [f"<{self.__class__.__name__}> '{self.name}'"]
        for j, item in enumerate(self.item_names):
            txt.append(f"- Item: {j}: {item}")
        return "\n".join(txt)
