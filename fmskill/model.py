import os
from typing import List, Union
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import warnings
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from mikeio import Dfs0, Dfsu, Dataset, eum
from .observation import Observation, PointObservation, TrackObservation
from .comparison import PointComparer, TrackComparer, ComparerCollection, BaseComparer
from .plot import plot_observation_positions
from .utils import make_unique_index


class ModelResultInterface(ABC):
    @property
    @abstractmethod
    def start_time(self):
        pass

    @property
    @abstractmethod
    def end_time(self):
        pass

    @property
    def itemInfo(self):
        return None


class DataFrameModelResult(ModelResultInterface):
    def __init__(self, df, name: str = None, item=None):
        if isinstance(df, pd.DataFrame):
            if item is None:
                if len(df.columns) == 1:
                    item = 0
                else:
                    raise ValueError("Model ambiguous - please provide item")

            if isinstance(item, str):
                df = df[[item]]
            elif isinstance(item, int):
                df = df.iloc[:, item].to_frame()
            else:
                raise TypeError("item must be int or string")

        elif isinstance(df, pd.Series):
            df = df.to_frame()
        self.df = df
        if name is None:
            name = self.df.columns[0]
        self.name = name

    @property
    def start_time(self):
        return self.df.index[0].to_pydatetime()

    @property
    def end_time(self):
        return self.df.index[-1].to_pydatetime()


class ModelResult(ModelResultInterface):
    """
    The result from a MIKE FM simulation (either dfsu or dfs0)

    Examples
    --------
    >>> mr = ModelResult("Oresund2D.dfsu")

    >>> mr = ModelResult("Oresund2D_points.dfs0", name="Oresund")
    """

    def __init__(self, filename: str, name: str = None, item=None):
        # TODO: add "start" as user may wish to disregard start from comparison
        self.filename = filename
        ext = os.path.splitext(filename)[-1]
        if ext == ".dfsu":
            self.dfs = Dfsu(filename)
        # elif ext == '.dfs2':
        #    self.dfs = Dfs2(filename)
        elif ext == ".dfs0":
            self.dfs = Dfs0(filename)
        else:
            raise ValueError(f"Filename extension {ext} not supported (dfsu, dfs0)")

        # self.observations = {}

        if name is None:
            name = os.path.basename(filename).split(".")[0]
        self.name = name
        self.item = self._parse_item(item)

    def _parse_item(self, item, items=None):
        if items is None:
            items = self.dfs.items
        n_items = len(items)
        if item is None:
            if n_items == 1:
                return 0
            else:
                return None
        if isinstance(item, int):
            if (item < 0) or (item >= n_items):
                raise ValueError(f"item must be between 0 and {n_items-1}")
        elif isinstance(item, str):
            item_names = [i.name for i in items]
            if item not in item_names:
                raise ValueError(f"item must be one of {item_names}")
            item = item_names.index(item)
        else:
            raise ValueError("item must be int or string")
        return item

    @property
    def start_time(self):
        return self.dfs.start_time

    @property
    def end_time(self):
        return self.dfs.end_time

    @property
    def itemInfo(self):
        if self.item is None:
            return eum.ItemInfo(eum.EUMType.Undefined)
        else:
            # if isinstance(self.item, str):
            self.item = self._parse_item(self.item)
            return self.dfs.items[self.item]

    def __repr__(self):
        out = []
        out.append("<fmskill.ModelResult>")
        out.append(self.filename)
        return "\n".join(out)

    @staticmethod
    def from_config(configuration: Union[dict, str], validate_eum=True):
        warnings.warn(
            "ModelResult.from_config is deprecated, use Connector instead",
            DeprecationWarning,
        )

    def add_observation(self, observation, item=None, weight=1.0, validate_eum=True):
        warnings.warn(
            "ModelResult.add_observation is deprecated, use Connector instead",
            DeprecationWarning,
        )

    def _in_domain(self, x, y) -> bool:
        ok = True
        if self.is_dfsu:
            ok = self.dfs.contains([x, y])
        return ok

    def _validate_observation(self, observation) -> bool:
        ok = False
        if self.is_dfsu:
            if isinstance(observation, PointObservation):
                ok = self.dfs.contains([observation.x, observation.y])
                if not ok:
                    raise ValueError("Observation outside domain")
            elif isinstance(observation, TrackObservation):
                ok = True
        elif self.is_dfs0:
            # TODO: add check on name
            ok = True
        if ok:
            ok = self._validate_start_end(observation)
            if not ok:
                warnings.warn("No time overlap between model result and observation!")
        return ok

    def _validate_start_end(self, observation: Observation) -> bool:
        try:
            # need to put this in try-catch due to error in dfs0 in mikeio
            if observation.end_time < self.dfs.start_time:
                return False
            if observation.start_time > self.dfs.end_time:
                return False
        except:
            pass
        return True

    def _validate_item_eum(
        self, observation: Observation, item, mod_items=None
    ) -> bool:
        """Check that observation and model item eum match"""
        if mod_items is None:
            mod_items = self.dfs.items
        ok = True
        obs_item = observation.itemInfo
        if obs_item.type == eum.EUMType.Undefined:
            warnings.warn(f"{observation.name}: Cannot validate as type is Undefined.")
            return ok

        item = self._get_model_item(item, mod_items)
        if item.type != obs_item.type:
            ok = False
            warnings.warn(
                f"{observation.name}: Item type should match. Model item: {item.type.display_name}, obs item: {obs_item.type.display_name}"
            )
        if item.unit != obs_item.unit:
            ok = False
            warnings.warn(
                f"{observation.name}: Unit should match. Model unit: {item.unit.display_name}, obs unit: {obs_item.unit.display_name}"
            )
        return ok

    def _get_model_item(self, item, mod_items=None) -> eum.ItemInfo:
        """Given str or int find corresponding model itemInfo"""
        if mod_items is None:
            mod_items = self.dfs.items
        n_items = len(mod_items)
        if isinstance(item, int):
            if (item < 0) or (item >= n_items):
                raise ValueError(f"item number must be between 0 and {n_items}")
        elif isinstance(item, str):
            item_names = [i.name for i in mod_items]
            try:
                item = item_names.index(item)
            except ValueError:
                raise ValueError(f"item not found in model items ({item_names})")
        else:
            raise ValueError("item must be an integer or a string")
        return mod_items[item]

    def _infer_model_item(
        self,
        observation: Observation,
        mod_items: List[eum.ItemInfo] = None,
    ) -> int:
        """Attempt to infer model item by matching observation eum with model eum"""
        if mod_items is None:
            mod_items = self.dfs.items

        if len(mod_items) == 1:
            # accept even if eum does not match
            return 0

        mod_items = [(x.type, x.unit) for x in mod_items]
        obs_item = (observation.itemInfo.type, observation.itemInfo.unit)

        pot_items = [j for j, mod_item in enumerate(mod_items) if mod_item == obs_item]

        if len(pot_items) == 0:
            raise Exception("Could not infer")
        if len(pot_items) > 1:
            raise ValueError(
                f"Multiple matching model items found! (Matches {pot_items})."
            )

        return pot_items[0]

    def extract(self) -> ComparerCollection:
        warnings.warn(
            "ModelResult.extract is deprecated, use Connector instead",
            DeprecationWarning,
        )

    def extract_observation(
        self,
        observation: Union[PointObservation, TrackObservation],
        item: Union[int, str] = None,
        validate: bool = True,
    ) -> BaseComparer:
        """Compare this ModelResult with an observation

        Parameters
        ----------
        observation : <PointObservation> or <TrackObservation>
            Observation to be compared
        item : str, integer
            ModelResult item name or number
            If None, then try to infer from observation eum value.
            Default: None
        validate: bool, optional
            Validate if observation is inside domain and that eum type
            and units; Defaut: True

        Returns
        -------
        <fmskill.BaseComparer>
            A comparer object for further analysis or plotting
        """
        if item is None:
            item = self.item
        if item is None:
            item = self._infer_model_item(observation)

        if validate:
            ok = self._validate_observation(observation)
            if ok:
                ok = self._validate_item_eum(observation, item)
            if not ok:
                raise ValueError("Could not extract observation")

        if isinstance(observation, PointObservation):
            df_model = self._extract_point(observation, item)
            comparer = PointComparer(observation, df_model)
        elif isinstance(observation, TrackObservation):
            df_model = self._extract_track(observation, item)
            comparer = TrackComparer(observation, df_model)
        else:
            raise ValueError("Only point and track observation are supported!")

        if len(comparer.df) == 0:
            warnings.warn(f"No overlapping data in found for {observation.name}!")
            comparer = None

        return comparer

    def _extract_point(self, observation: PointObservation, item) -> pd.DataFrame:
        ds_model = None
        if self.is_dfsu:
            ds_model = self._extract_point_dfsu(observation.x, observation.y, item)
        elif self.is_dfs0:
            ds_model = self._extract_point_dfs0(item)

        return ds_model.to_dataframe()

    def _extract_point_dfsu(self, x, y, item) -> Dataset:
        xy = np.atleast_2d([x, y])
        elemids, _ = self.dfs.get_2d_interpolant(xy, n_nearest=1)
        ds_model = self.dfs.read(elements=elemids, items=[item])
        ds_model.items[0].name = self.name
        return ds_model

    def _extract_point_dfs0(self, item=None) -> Dataset:
        if item is None:
            item = self.item
        ds_model = self.dfs.read(items=[item])
        ds_model.items[0].name = self.name
        return ds_model

    def _extract_track(self, observation: TrackObservation, item) -> pd.DataFrame:
        df = None
        if self.is_dfsu:
            ds_model = self._extract_track_dfsu(observation, item)
            df = ds_model.to_dataframe().dropna()
        elif self.is_dfs0:
            ds_model = self.dfs.read(items=[0, 1, item])
            ds_model.items[-1].name = self.name
            df = ds_model.to_dataframe().dropna()
            df.index = make_unique_index(df.index, offset_in_seconds=0.01)
        return df

    def _extract_track_dfsu(self, observation: TrackObservation, item) -> Dataset:
        ds_model = self.dfs.extract_track(track=observation.df, items=[item])
        ds_model.items[-1].name = self.name
        return ds_model

    def plot_observation_positions(self, figsize=None):
        warnings.warn(
            "ModelResult.plot_observation_positions is deprecated, use Connector instead",
            DeprecationWarning,
        )

    def plot_temporal_coverage(self, limit_to_model_period=True):
        warnings.warn(
            "ModelResult.plot_temporal_coverage is deprecated, use Connector instead",
            DeprecationWarning,
        )

    @property
    def is_dfsu(self):
        return isinstance(self.dfs, Dfsu)

    @property
    def is_dfs0(self):
        return isinstance(self.dfs, Dfs0)
