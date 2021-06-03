import os
from typing import Union
import numpy as np
import pandas as pd
import warnings

from mikeio import Dfs0, Dfsu, Dataset, eum
from ..observation import Observation, PointObservation, TrackObservation
from ..comparison import PointComparer, TrackComparer, ComparerCollection, BaseComparer
from ..utils import make_unique_index
from .abstract import ModelResultInterface, MultiItemModelResult


class _DfsBase:
    dfs = None
    name = None
    _filename = None
    _selected_item = None

    @property
    def start_time(self):
        return self.dfs.start_time

    @property
    def end_time(self):
        return self.dfs.end_time

    @property
    def filename(self):
        return self._filename

    @property
    def is_dfsu(self):
        return isinstance(self.dfs, Dfsu)

    @property
    def is_dfs0(self):
        return isinstance(self.dfs, Dfs0)

    def _in_domain(self, x, y) -> bool:
        ok = True
        if self.is_dfsu:
            ok = self.dfs.contains([x, y])
        return ok

    def _get_item_num(self, item) -> int:
        items = self.dfs.items
        n_items = len(items)
        if item is None:
            if n_items == 1:
                return 0
            else:
                return None
        if isinstance(item, eum.ItemInfo):
            item = item.name
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

    def _get_item_name(self, item) -> str:
        item_num = self._get_item_num(item)
        return self.dfs.items[item_num].name

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

    def _extract_point(self, observation: PointObservation, item=None) -> pd.DataFrame:
        if item is None:
            item = self._selected_item
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

    def _extract_point_dfs0(self, item) -> Dataset:
        ds_model = self.dfs.read(items=[item])
        ds_model.items[0].name = self.name
        return ds_model

    def _extract_track(self, observation: TrackObservation, item=None) -> pd.DataFrame:
        if item is None:
            item = self._selected_item
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

    def _deprecation_message(self, method: str):
        warnings.warn(
            f"{method} is deprecated from v.0.3, use Connector instead",
            DeprecationWarning,
        )

    @staticmethod
    def from_config(configuration: Union[dict, str], validate_eum=True):
        self._deprecation_message("ModelResult.from_config()")
        raise NotImplementedError("Deprecated! Use Connector instead.")

    def add_observation(self, observation, item=None, weight=1.0, validate_eum=True):
        self._deprecation_message("ModelResult.add_observation()")
        raise NotImplementedError("Deprecated! Use Connector instead.")

    def extract(self) -> ComparerCollection:
        self._deprecation_message("ModelResult.extract()")
        raise NotImplementedError("Deprecated! Use Connector instead.")


class DfsModelResultItem(_DfsBase, ModelResultInterface):
    @property
    def item_name(self):
        return self.itemInfo.name

    def __init__(self, dfs, itemInfo, filename, name):
        self.dfs = dfs
        self.itemInfo = itemInfo
        self._selected_item = self._get_item_num(itemInfo)
        self._filename = filename
        self.name = name

    def __repr__(self):
        txt = [f"<DfsModelResultItem> '{self.name}'"]
        txt.append(f"File: {self.filename}")
        item_num = self._get_item_num(self.item_name)
        txt.append(f"- Item: {item_num}: {self.itemInfo}")
        return "\n".join(txt)

    def extract_observation(
        self, observation: Union[PointObservation, TrackObservation], validate=True
    ) -> BaseComparer:
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
        <fmskill.BaseComparer>
            A comparer object for further analysis or plotting
        """
        # if item is None:
        item = self._selected_item
        # if item is None:
        #     item = self._infer_model_item(observation)

        if validate:
            ok = self._validate_observation(observation)
            if ok:
                ok = self._validate_item_eum(observation)
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
            warnings.warn(f"No overlapping data in found for obs '{observation.name}'!")
            comparer = None

        return comparer

    def _validate_item_eum(self, observation: Observation) -> bool:
        """Check that observation and model item eum match"""
        ok = True
        obs_item = observation.itemInfo
        if obs_item.type == eum.EUMType.Undefined:
            warnings.warn(f"{observation.name}: Cannot validate as type is Undefined.")
            return ok

        item = self.itemInfo
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


class DfsModelResult(_DfsBase, MultiItemModelResult):
    @property
    def item_names(self):
        return [item.name for item in self.dfs.items]

    @property
    def n_items(self):
        return len(self.dfs.items)

    def __init__(self, filename: str, name: str = None, item=None):
        # TODO: add "start" as user may wish to disregard start from comparison
        self._filename = filename
        ext = os.path.splitext(filename)[-1]
        if ext == ".dfsu":
            self.dfs = Dfsu(filename)
        # elif ext == '.dfs2':
        #    self.dfs = Dfs2(filename)
        elif ext == ".dfs0":
            self.dfs = Dfs0(filename)
        else:
            raise ValueError(f"Filename extension {ext} not supported (dfsu, dfs0)")

        if name is None:
            name = os.path.basename(filename).split(".")[0]
        self.name = name

        self._mr_items = {}
        for it in self.dfs.items:
            self._mr_items[it.name] = DfsModelResultItem(
                self.dfs, it, self._filename, self.name
            )

        if item is not None:
            self._selected_item = self._get_item_num(item)
        elif len(self._mr_items) == 1:
            self._selected_item = 0
        else:
            self._selected_item = None

    def __repr__(self):
        txt = [f"<DfsModelResult> '{self.name}'"]
        txt.append(f"File: {self.filename}")
        for j, item in enumerate(self.dfs.items):
            txt.append(f"- Item: {j}: {item}")
        return "\n".join(txt)


#     def _infer_model_item(
#         self,
#         observation: Observation,
#         mod_items: List[eum.ItemInfo] = None,
#     ) -> int:
#         """Attempt to infer model item by matching observation eum with model eum"""
#         if mod_items is None:
#             mod_items = self.dfs.items

#         if len(mod_items) == 1:
#             # accept even if eum does not match
#             return 0

#         mod_items = [(x.type, x.unit) for x in mod_items]
#         obs_item = (observation.itemInfo.type, observation.itemInfo.unit)

#         pot_items = [j for j, mod_item in enumerate(mod_items) if mod_item == obs_item]

#         if len(pot_items) == 0:
#             raise Exception("Could not infer")
#         if len(pot_items) > 1:
#             raise ValueError(
#                 f"Multiple matching model items found! (Matches {pot_items})."
#             )

#         return pot_items[0]
