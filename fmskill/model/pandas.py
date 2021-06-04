import pandas as pd
import warnings

from mikeio import eum
from ..observation import PointObservation
from ..comparison import PointComparer
from .abstract import ModelResultInterface, MultiItemModelResult


class _DataFrameBase:
    df = None
    itemInfo = eum.ItemInfo(eum.EUMType.Undefined)

    @property
    def start_time(self):
        return self.df.index[0].to_pydatetime()

    @property
    def end_time(self):
        return self.df.index[-1].to_pydatetime()

    def _get_item_name(self, item, item_names=None) -> str:
        if item_names is None:
            item_names = list(self.df.columns)
        n_items = len(item_names)
        if item is None:
            if n_items == 1:
                return item_names[0]
            else:
                return None
        if isinstance(item, eum.ItemInfo):
            item = item.name
        if isinstance(item, int):
            if (item < 0) or (item >= n_items):
                raise ValueError(f"item must be between 0 and {n_items-1}")
            item = item_names[item]
        elif isinstance(item, str):
            if item not in item_names:
                raise ValueError(f"item must be one of {item_names}")
        else:
            raise ValueError("item must be int or string")
        return item

    def _get_item_num(self, item) -> int:
        item_name = self._get_item_name(item)
        item_names = list(self.df.columns)
        return item_names.index(item_name)


class DataFrameModelResultItem(_DataFrameBase, ModelResultInterface):
    @property
    def item_name(self):
        return self.df.columns[0]

    def __init__(self, df, name: str = None, item=None):
        if isinstance(df, (pd.Series, pd.DataFrame)):
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TypeError("Index must be DatetimeIndex!")
        else:
            raise TypeError("Input must be pandas Series or DataFrame!")

        if isinstance(df, pd.Series):
            df = df.to_frame()
            df.columns = ["model"] if name is None else name

        if item is None:
            if len(df.columns) == 1:
                item = df.columns[0]
            else:
                raise ValueError("Model ambiguous - please provide item")
        item = self._get_item_name(item, df.columns)
        self.df = df[[item]]
        self._selected_item = item
        if name is None:
            name = self.item_name
        self.name = name

    def __repr__(self):
        txt = [f"<DataFrameModelResultItem> '{self.name}'"]
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
        if isinstance(observation, PointObservation):
            comparer = PointComparer(observation, self.df)
        else:
            raise ValueError("Only point observation are supported!")

        if len(comparer.df) == 0:
            warnings.warn(f"No overlapping data in found for obs '{observation.name}'!")
            comparer = None

        return comparer


class DataFrameModelResult(_DataFrameBase, MultiItemModelResult):
    @property
    def item_names(self):
        return list(self.df.columns)

    def __init__(self, df, name: str = None, item=None):
        if isinstance(df, (pd.Series, pd.DataFrame)):
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TypeError("Index must be DatetimeIndex!")
        else:
            raise TypeError("Input must be pandas Series or DataFrame!")

        if isinstance(df, pd.Series):
            df = df.to_frame()
            df.columns = ["model"] if name is None else name
        self.df = df

        self._selected_item = None
        if item is None:
            if len(df.columns) == 1:
                self._selected_item = df.columns[0]
        else:
            self._selected_item = self._get_item_name(item)

        if name is None:
            if self._selected_item is None:
                name = "model"
            else:
                name = self.df.columns[self._selected_item]
        self.name = name

        self._mr_items = {}
        for it in self.item_names:
            self._mr_items[it] = DataFrameModelResultItem(self.df, self.name, it)

    def __repr__(self):
        txt = [f"<DataFrameModelResult> '{self.name}'"]
        for j, item in enumerate(self.item_names):
            txt.append(f"- Item: {j}: {item}")
        return "\n".join(txt)
