import os
from typing import Union
import numpy as np
import pandas as pd
import warnings

import mikeio
from ..observation import Observation, PointObservation, TrackObservation
from ..comparison import (
    PointComparer,
    TrackComparer,
    ComparerCollection,
    SingleObsComparer,
)
from ..utils import make_unique_index
from .legacy_abstract import ModelResultInterface


def _validate_item_eum(mod_item: mikeio.ItemInfo, obs: Observation) -> bool:
    """Check that observation and model item eum match"""
    ok = True
    if obs.itemInfo.type == mikeio.EUMType.Undefined:
        warnings.warn(f"{obs.name}: Cannot validate as type is Undefined.")
        return ok

    if mod_item.type != obs.itemInfo.type:
        ok = False
        warnings.warn(
            f"{obs.name}: Item type should match. Model item: {mod_item.type.display_name}, obs item: {obs.itemInfo.type.display_name}"
        )
    if mod_item.unit != obs.itemInfo.unit:
        ok = False
        warnings.warn(
            f"{obs.name}: Unit should match. Model unit: {mod_item.unit.display_name}, obs unit: {obs.itemInfo.unit.display_name}"
        )
    return ok


class _DfsBase:
    @property
    def start_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.data.start_time)

    @property
    def end_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.data.end_time)

    @property
    def filename(self):
        return self._filename

    @property
    def is_dfsu(self):
        return isinstance(self.data, mikeio.dfsu._Dfsu)

    @property
    def is_dfs0(self):
        return isinstance(self.data, mikeio.Dfs0)

    def _in_domain(self, x, y) -> bool:
        ok = True
        if self.is_dfsu:
            ok = self.data.contains([x, y])
        return ok

    def _get_item_num(self, item) -> int:
        items = self.data.items
        n_items = len(items)
        if item is None:
            if n_items == 1:
                return 0
            else:
                return None
        if isinstance(item, mikeio.ItemInfo):
            item = item.name
        if isinstance(item, int):
            if item < 0:  # Handle negative indices
                item = n_items + item
            if (item < 0) or (item >= n_items):
                raise IndexError(f"item {item} out of range (0, {n_items-1})")
        elif isinstance(item, str):
            item_names = [i.name for i in items]
            if item not in item_names:
                raise KeyError(f"item must be one of {item_names}")
            item = item_names.index(item)
        else:
            raise TypeError("item must be int or string")
        return item

    def _get_item_name(self, item) -> str:
        item_num = self._get_item_num(item)
        return self.data.items[item_num].name

    def _validate_observation(self, observation) -> bool:
        ok = False
        if self.is_dfsu:
            if isinstance(observation, PointObservation):
                ok = self.data.contains([observation.x, observation.y])
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
            if observation.end_time < self.data.start_time:
                return False
            if observation.start_time > self.data.end_time:
                return False
        except:
            pass
        return True

    def _extract_point(self, observation: PointObservation, item=None) -> pd.DataFrame:
        if item is None:
            item = self._selected_item
        ds_model = None
        if self.is_dfsu:
            if (observation.x is None) or (observation.y is None):
                raise ValueError(
                    f"PointObservation '{observation.name}' cannot be used for extraction "
                    + f"because it has None position x={observation.x}, y={observation.y}. "
                    + "Please provide position when creating PointObservation."
                )
            ds_model = self._extract_point_dfsu(observation.x, observation.y, item)
        elif self.is_dfs0:
            ds_model = self._extract_point_dfs0(item)

        return ds_model.to_dataframe().dropna()

    def _extract_point_dfsu(self, x, y, item) -> mikeio.Dataset:
        xy = np.atleast_2d([x, y])
        elemids = self.data.geometry.find_index(coords=xy)
        ds_model = self.data.read(elements=elemids, items=[item])
        ds_model.rename({ds_model.items[0].name: self.name}, inplace=True)
        return ds_model

    def _extract_point_dfs0(self, item) -> mikeio.Dataset:
        ds_model = self.data.read(items=[item])
        ds_model.rename({ds_model.items[0].name: self.name}, inplace=True)
        return ds_model

    def _extract_track(self, observation: TrackObservation, item=None) -> pd.DataFrame:
        if item is None:
            item = self._selected_item
        df = None
        if self.is_dfsu:
            ds_model = self._extract_track_dfsu(observation, item)
            df = ds_model.to_dataframe().dropna()
        elif self.is_dfs0:
            ds_model = self.data.read(items=[0, 1, item])
            ds_model.rename({ds_model.items[-1].name: self.name}, inplace=True)
            df = ds_model.to_dataframe().dropna()
            df.index = make_unique_index(df.index, offset_duplicates=0.001)
        return df

    def _extract_track_dfsu(
        self, observation: TrackObservation, item
    ) -> mikeio.Dataset:
        ds_model = self.data.extract_track(track=observation.data, items=[item])
        ds_model.rename({ds_model.items[-1].name: self.name}, inplace=True)
        return ds_model

    @staticmethod
    def _deprecation_message(self, method: str):
        warnings.warn(
            f"{method} is deprecated from v.0.3, use Connector instead",
            DeprecationWarning,
        )

    @staticmethod
    def from_config(configuration: Union[dict, str], validate_eum=True):
        _DfsBase._deprecation_message("ModelResult.from_config()")
        raise NotImplementedError("Deprecated! Use Connector instead.")

    def add_observation(self, observation, item=None, weight=1.0, validate_eum=True):
        self._deprecation_message("ModelResult.add_observation()")
        raise NotImplementedError("Deprecated! Use Connector instead.")

    def extract(self) -> ComparerCollection:
        self._deprecation_message("ModelResult.extract()")
        raise NotImplementedError("Deprecated! Use Connector instead.")


class DataArrayModelResultItem(ModelResultInterface):
    @property
    def start_time(self) -> pd.Timestamp:
        return self.data.time[0]

    @property
    def end_time(self) -> pd.Timestamp:
        return self.data.time[-1]

    @property
    def item_name(self):
        return self.data.name

    def __init__(self, da: mikeio.DataArray, name=None):
        self.data = da
        if name is None:
            self.name = self.data.name
        else:
            self.name = name

    def __repr__(self):
        txt = [f"<DataArrayModelResultItem> '{self.name}'"]
        txt.append(f"- Item: {self.itemInfo}")
        return "\n".join(txt)

    @property
    def itemInfo(self):
        return self.data.item

    def _extract_point(self, observation: PointObservation) -> pd.DataFrame:

        dap = self.data.sel(x=observation.x, y=observation.y)
        dap.name = self.name
        # ds_model.rename({ds_model.items[0].name: self.name}, inplace=True)

        # Why is there no .to_dataframe() on DataArray?
        return mikeio.Dataset(dap).to_dataframe().dropna()

    def _extract_track(self, observation: TrackObservation) -> pd.DataFrame:
        ds = self.data.extract_track(observation.data)
        ds.rename({ds.items[-1].name: self.name}, inplace=True)
        return ds.to_dataframe().dropna()

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
            ok = _validate_item_eum(self.itemInfo, observation)
            if not ok:
                raise ValueError("Could not extract observation")

        if isinstance(observation, PointObservation):
            df_model = self._extract_point(observation)
            comparer = PointComparer(observation, df_model)
        elif isinstance(observation, TrackObservation):
            df_model = self._extract_track(observation)
            comparer = TrackComparer(observation, df_model)
        else:
            raise ValueError("Only point and track observation are supported!")

        if len(comparer.data) == 0:
            warnings.warn(f"No overlapping data in found for obs '{observation.name}'!")
            comparer = None

        return comparer


class DfsModelResultItem(_DfsBase, ModelResultInterface):
    @property
    def item_name(self):
        return self.itemInfo.name

    def __init__(self, filename: str, name: str = None, item=None, itemInfo=None):
        # TODO: add "start" as user may wish to disregard start from comparison
        self._filename = filename
        ext = os.path.splitext(filename)[-1]
        if ext == ".dfsu":
            self.data = mikeio.open(filename)
        # elif ext == '.dfs2':
        #    self.data = mikeio.open(filename)
        elif ext == ".dfs0":
            self.data = mikeio.open(filename)
        else:
            raise ValueError(f"Filename extension {ext} not supported (dfsu, dfs0)")

        if name is None:
            name = os.path.basename(filename).split(".")[0]
        self.name = name

        if item is not None:
            self._selected_item = self._get_item_num(item)
        elif len(self.data.items) == 1:
            self._selected_item = 0
        else:
            item_names = [i.name for i in self.data.items]
            raise ValueError(
                f"Dfs file has multiple items: item must be specified! Available items: {item_names}"
            )

        self.itemInfo = (
            self.data.items[self._selected_item] if itemInfo is None else itemInfo
        )

    def __repr__(self):
        txt = [f"<DfsModelResultItem> '{self.name}'"]
        txt.append(f"File: {self.filename}")
        item_num = self._get_item_num(self.item_name)
        txt.append(f"- Item: {item_num}: {self.itemInfo}")
        return "\n".join(txt)

    def extract_observation(
        self,
        observation: Union[PointObservation, TrackObservation],
        validate=True,
        **kwargs,
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
        # if item is None:
        item = self._selected_item
        # if item is None:
        #     item = self._infer_model_item(observation)

        if validate:
            ok = self._validate_observation(observation)
            if ok:
                ok = _validate_item_eum(self.itemInfo, observation)
            if not ok:
                raise ValueError("Could not extract observation")

        if isinstance(observation, PointObservation):
            df_model = self._extract_point(observation, item)
            comparer = PointComparer(observation, df_model, **kwargs)
        elif isinstance(observation, TrackObservation):
            df_model = self._extract_track(observation, item)
            comparer = TrackComparer(observation, df_model, **kwargs)
        else:
            raise ValueError("Only point and track observation are supported!")

        if len(comparer.data) == 0:
            warnings.warn(f"No overlapping data in found for obs '{observation.name}'!")
            comparer = None

        return comparer