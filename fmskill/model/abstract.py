from abc import ABC, abstractmethod
import pandas as pd
import mikeio

from ..comparison import SingleObsComparer


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

    @abstractmethod
    def extract_observation(self, observation) -> SingleObsComparer:
        pass

    def __repr__(self):
        txt = [f"<{self.__class__.__name__}> '{self.name}'"]
        txt.append(f"- Item: {self.item_name}")
        return "\n".join(txt)

    def _in_domain(self, x, y) -> bool:
        return True
