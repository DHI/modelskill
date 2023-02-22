from fmskill.utils import make_unique_index
import pandas as pd
import warnings

import mikeio
from ..observation import PointObservation, TrackObservation
from ..comparison import PointComparer, TrackComparer
from .abstract import ModelResultInterface, _parse_itemInfo


class _DataFrameBase:
    @property
    def start_time(self) -> pd.Timestamp:
        return self.data.index[0]

    @property
    def end_time(self) -> pd.Timestamp:
        return self.data.index[-1]

    @staticmethod
    def _check_dataframe(df):
        if isinstance(df, pd.DataFrame):
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TypeError("Index must be DatetimeIndex!")
        else:
            raise TypeError("Input must be pandas DataFrame!")

    def _get_selected_item(self, cols, item):
        sel_item = None
        if item is None:
            if len(cols) == 1:
                sel_item = cols[0]
        else:
            sel_item = self._get_item_name(item)
        return sel_item

    @staticmethod
    def _get_default_item(cols):
        if len(cols) == 1:
            return cols[0]
        else:
            raise ValueError(
                f"Model ambiguous - please provide item (=column name). Available columns: {list(cols)}"
            )

    def _get_item_name(self, item, item_names=None) -> str:
        if item_names is None:
            item_names = list(self.data.columns)
        n_items = len(item_names)
        if item is None:
            if n_items == 1:
                return item_names[0]
            else:
                return None
        if isinstance(item, mikeio.ItemInfo):
            item = item.name
        elif isinstance(item, int):
            if item < 0:  # Handle negative indices
                item = n_items + item
            if (item < 0) or (item >= n_items):
                raise IndexError(f"item {item} out of range (0, {n_items-1})")
            item = item_names[item]
        elif isinstance(item, str):
            if item not in item_names:
                raise KeyError(f"item must be one of {item_names}")
        else:
            raise TypeError("item must be int or string")
        return item

    def _get_item_num(self, item, item_names=None) -> int:
        item_name = self._get_item_name(item, item_names)
        if item_names is None:
            item_names = list(self.data.columns)
        return item_names.index(item_name)

    def _extract_point(self, observation: PointObservation, item=None) -> pd.DataFrame:
        assert isinstance(self, DataFramePointModelResultItem)
        if item is None:
            item = self._selected_item
        else:
            item = self._get_item_name(item)
        return self.data[[item]].dropna()


class DataFramePointModelResultItem(_DataFrameBase, ModelResultInterface):
    @property
    def item_name(self):
        return self.data.columns[0]

    def __init__(self, df, name: str = None, item=None, itemInfo=None):
        self.itemInfo = _parse_itemInfo(itemInfo)
        self.is_point = True
        if isinstance(df, pd.Series):
            df = df.to_frame()
            df.columns = ["model"] if name is None else name
        self._check_dataframe(df)

        if item is None:
            item = self._get_default_item(df.columns)

        item = self._get_item_name(item, df.columns)
        self.data = df[[item]]
        self._selected_item = item

        if name is None:
            name = self.item_name
        self.name = name

    def extract_observation(
        self, observation: PointObservation, **kwargs
    ) -> PointComparer:
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
            comparer = PointComparer(observation, df_model, **kwargs)
        else:
            raise ValueError("Only point observation are supported!")

        if len(comparer.df) == 0:
            warnings.warn(f"No overlapping data in found for obs '{observation.name}'!")
            comparer = None

        return comparer


class DataFrameTrackModelResultItem(_DataFrameBase, ModelResultInterface):
    @property
    def item_name(self):
        return self._selected_item

    def __init__(
        self,
        df,
        name: str = None,
        item=None,
        itemInfo=None,
        x_item=None,
        y_item=None,
    ):
        self.itemInfo = _parse_itemInfo(itemInfo)
        self.is_point = False
        self._x_item, self._y_item = self._parse_x_y_columns(df, x_item, y_item)
        self._check_dataframe(df)
        if item is None:
            val_cols = self._get_val_cols(df.columns)
            item = self._get_default_item(val_cols)

        if not df.index.is_unique:
            df = df.copy()
            df.index = make_unique_index(df.index)

        item_num = self._get_item_num(item, list(df.columns))
        self.data = df.iloc[:, [self._x_item, self._y_item, item_num]]
        item = self._get_item_name(item, self.data.columns)
        self._selected_item = item

        if name is None:
            name = self.item_name
        self.name = name

    def _parse_x_y_columns(self, df, x, y):
        if x is None:
            x = self._get_default_x_idx(df.columns[0])
        else:
            x = self._get_col_idx(df.columns, x)

        if y is None:
            y = self._get_default_y_idx(df.columns[1])
        else:
            y = self._get_col_idx(df.columns, y)
        return x, y

    @staticmethod
    def _get_default_x_idx(n):
        n = n.lower()
        if (n == "x") or ("lon" in n) or ("east" in n):
            return 0
        else:
            raise ValueError(
                f"First column '{n}' does not seem like a plausible x-coordinate column name. Please provide explicitly using the 'x_item' keyword."
            )

    @staticmethod
    def _get_default_y_idx(n):
        n = n.lower()
        if (n == "y") or ("lat" in n) or ("north" in n):
            return 1
        else:
            raise ValueError(
                f"Second column '{n}' does not seem like a plausible y-coordinate column name. Please provide explicitly using the 'y_item' keyword."
            )

    @staticmethod
    def _get_col_idx(cols, col):
        if isinstance(col, str):
            col = list(cols).index(col)
        elif isinstance(col, int):
            if col < 0:
                col = len(cols) + col
            if (col < 0) or (col >= len(cols)):
                raise ValueError(
                    f"column must be between 0 and {len(cols)-1} (or {-len(cols)} and -1)"
                )
        else:
            raise TypeError("column must be given as int or str")
        return col

    @property
    def _val_cols(self):
        """All columns except x- and y- column"""
        return self._get_val_cols(self.data.columns)

    def _get_val_cols(self, cols):
        """All columns except x- and y- column"""
        col_ids = [
            j for j in range(len(cols)) if (j != self._x_item and j != self._y_item)
        ]
        return cols[col_ids]

    def _extract_track(self, observation: TrackObservation, item=None) -> pd.DataFrame:
        assert isinstance(self, DataFrameTrackModelResultItem)
        if item is None:
            item = self._selected_item
        item_num = self._get_item_num(item)
        return self.data.iloc[:, [self._x_item, self._y_item, item_num]].dropna()

    def extract_observation(
        self, observation: TrackObservation, **kwargs
    ) -> TrackComparer:
        """Compare this ModelResult with an observation

        Parameters
        ----------
        observation : <TrackObservation>
            Observation to be compared

        Returns
        -------
        <fmskill.TrackComparer>
            A comparer object for further analysis or plotting
        """
        if isinstance(observation, TrackObservation):
            df_model = self._extract_track(observation)
            comparer = TrackComparer(observation, df_model, **kwargs)
        else:
            raise ValueError("Only track observation are supported!")

        if len(comparer.df) == 0:
            warnings.warn(f"No overlapping data in found for obs '{observation.name}'!")
            comparer = None

        return comparer
