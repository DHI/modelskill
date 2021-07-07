from fmskill.utils import make_unique_index
import pandas as pd
import warnings

from mikeio import eum
from ..observation import PointObservation, TrackObservation
from ..comparison import PointComparer, TrackComparer
from .abstract import ModelResultInterface, MultiItemModelResult


class _DataFrameBase:
    def __init__(self) -> None:
        self.df = None
        self.itemInfo = eum.ItemInfo(eum.EUMType.Undefined)
        self.is_point = True

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

    def _get_item_num(self, item, item_names=None) -> int:
        item_name = self._get_item_name(item, item_names)
        if item_names is None:
            item_names = list(self.df.columns)
        return item_names.index(item_name)

    def _extract_point(self, observation: PointObservation, item=None) -> pd.DataFrame:
        if item is None:
            item = self._selected_item
        else:
            item = self._get_item_name(item)
        return self.df[[item]]

    def _extract_track(self, observation: TrackObservation, item=None) -> pd.DataFrame:
        if item is None:
            item = self._selected_item
        item_num = self._get_item_num(item)
        return self.df.iloc[:, [0, 1, item_num]]


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
            df_model = self._extract_point(observation)
            comparer = PointComparer(observation, df_model)
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
                name = self._selected_item
        self.name = name

        self._mr_items = {}
        for it in self.item_names:
            self._mr_items[it] = DataFrameModelResultItem(self.df, self.name, it)

    def __repr__(self):
        txt = [f"<DataFrameModelResult> '{self.name}'"]
        for j, item in enumerate(self.item_names):
            txt.append(f"- Item: {j}: {item}")
        return "\n".join(txt)


class DataFrameTrackModelResultItem(_DataFrameBase, ModelResultInterface):
    @property
    def item_name(self):
        return self.df.columns[-1]

    def __init__(self, df, name: str = None, item=None):
        if isinstance(df, pd.DataFrame):
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TypeError("Index must be DatetimeIndex!")
        else:
            raise TypeError("Input must be pandas DataFrame!")

        if item is None:
            if len(df.columns) == 3:
                item = df.columns[-1]
            else:
                raise ValueError("Model ambiguous - please provide item")

        if not df.index.is_unique:
            df.index = make_unique_index(df.index)

        item_num = self._get_item_num(item, list(df.columns))
        self.df = df.iloc[:, [0, 1, item_num]]
        item = self._get_item_name(item, df.columns)
        self._selected_item = item

        if name is None:
            name = self.item_name
        self.name = name

    def __repr__(self):
        txt = [f"<DataFrameTrackModelResultItem> '{self.name}'"]
        txt.append(f"- Item: {self.item_name}")
        return "\n".join(txt)

    def extract_observation(self, observation: TrackObservation) -> TrackComparer:
        """Compare this ModelResult with an observation

        Parameters
        ----------
        observation : <TrackObservation>
            Observation to be compared

        Returns
        -------
        <fmskill.PointComparer>
            A comparer object for further analysis or plotting
        """
        if isinstance(observation, TrackObservation):
            df_model = self._extract_track(observation)
            comparer = TrackComparer(observation, df_model)
        else:
            raise ValueError("Only track observation are supported!")

        if len(comparer.df) == 0:
            warnings.warn(f"No overlapping data in found for obs '{observation.name}'!")
            comparer = None

        return comparer


class DataFrameTrackModelResult(_DataFrameBase, MultiItemModelResult):
    @property
    def item_names(self):
        return list(self.df.columns[2:])

    def __init__(self, df, name: str = None, item=None):
        if isinstance(df, pd.DataFrame):
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TypeError("Index must be DatetimeIndex!")
        else:
            raise TypeError("Input must be pandas DataFrame!")

        if not df.index.is_unique:
            df.index = make_unique_index(df.index)

        self.df = df

        self._selected_item = None
        if item is None:
            if len(df.columns) == 3:
                self._selected_item = df.columns[-1]
        else:
            self._selected_item = self._get_item_name(item)

        if name is None:
            if self._selected_item is None:
                name = "model"
            else:
                name = self._selected_item
        self.name = name

        self._mr_items = {}
        for it in self.item_names:
            self._mr_items[it] = DataFrameTrackModelResultItem(self.df, self.name, it)

    def __repr__(self):
        txt = [f"<DataFrameTrackModelResult> '{self.name}'"]
        for j, item in enumerate(self.item_names):
            txt.append(f"- Item: {j}: {item}")
        return "\n".join(txt)
