import os
import pandas as pd
import xarray as xr

from .dfs import DfsModelResult
from .pandas import DataFrameModelResult


class ModelResult:
    """
    ModelResult factory returning a specialized ModelResult object
    depending on the input.

    * dfs0 or dfsu file
    * pandas.DataFrame/Series
    * NetCDF/Grib: Under development!

    Note
    ----
    If an input has more than one item and the desired item is not
    specified as argument on construction, then the item of the
    modelresult 'mr' **must** be specified by e.g. mr[0] or mr['item_B']
    before connecting to an observation.

    Examples
    --------
    >>> mr = ModelResult("Oresund2D.dfsu")
    >>> mr_item = mr["Surface elevation"]
    >>> mr = ModelResult("Oresund2D_points.dfs0", name="Oresund")
    >>> mr_item = mr[0]
    >>> mr_item = ModelResult("Oresund2D.dfsu", item=0)
    >>> mr_item = ModelResult("Oresund2D.dfsu", item="Surface elevation")

    >>> mr = ModelResult(df)
    >>> mr = mr["Water Level"]
    >>> mr_item = ModelResult(df, item="Water Level")
    """

    def __new__(self, input, *args, **kwargs):
        if isinstance(input, str):
            filename = input
            ext = os.path.splitext(filename)[-1]
            if "dfs" in ext:
                mr = DfsModelResult(filename, *args, **kwargs)
                if mr._selected_item is not None:
                    return mr[mr._selected_item]
                else:
                    return mr
            else:
                # return XrModelResult(filename, *args, **kwargs)
                raise NotImplementedError()
        elif isinstance(input, (pd.DataFrame, pd.Series)):
            mr = DataFrameModelResult(input, *args, **kwargs)
            if mr._selected_item is not None:
                return mr[mr._selected_item]
            else:
                return mr
        elif isinstance(input, (xr.Dataset, xr.DataArray)):
            raise NotImplementedError()
            # return XrModelResult(input, *args, **kwargs)
        else:
            raise ValueError("Input type not supported (filename or DataFrame)")


# class ModelResult(ModelResultInterface):
#     """
#     The result from a MIKE FM simulation (either dfsu or dfs0)

#     Examples
#     --------
#     >>> mr = ModelResult("Oresund2D.dfsu")

#     >>> mr = ModelResult("Oresund2D_points.dfs0", name="Oresund")
#     """

#     def __init__(self, filename: str, name: str = None, item=None):
#         # TODO: add "start" as user may wish to disregard start from comparison
#         self.filename = filename
#         ext = os.path.splitext(filename)[-1]
#         if ext == ".dfsu":
#             self.dfs = Dfsu(filename)
#         # elif ext == '.dfs2':
#         #    self.dfs = Dfs2(filename)
#         elif ext == ".dfs0":
#             self.dfs = Dfs0(filename)
#         else:
#             raise ValueError(f"Filename extension {ext} not supported (dfsu, dfs0)")

#         # self.observations = {}

#         if name is None:
#             name = os.path.basename(filename).split(".")[0]
#         self.name = name
#         self.item = self._parse_item(item)

#     def _parse_item(self, item, items=None):
#         if items is None:
#             items = self.dfs.items
#         n_items = len(items)
#         if item is None:
#             if n_items == 1:
#                 return 0
#             else:
#                 return None
#         if isinstance(item, int):
#             if (item < 0) or (item >= n_items):
#                 raise ValueError(f"item must be between 0 and {n_items-1}")
#         elif isinstance(item, str):
#             item_names = [i.name for i in items]
#             if item not in item_names:
#                 raise ValueError(f"item must be one of {item_names}")
#             item = item_names.index(item)
#         else:
#             raise ValueError("item must be int or string")
#         return item

#     @property
#     def itemInfo(self):
#         if self.item is None:
#             return eum.ItemInfo(eum.EUMType.Undefined)
#         else:
#             # if isinstance(self.item, str):
#             self.item = self._parse_item(self.item)
#             return self.dfs.items[self.item]

#     def __repr__(self):
#         out = []
#         out.append("<fmskill.ModelResult>")
#         out.append(self.filename)
#         return "\n".join(out)

#     @staticmethod
#     def from_config(configuration: Union[dict, str], validate_eum=True):
#         warnings.warn(
#             "ModelResult.from_config is deprecated, use Connector instead",
#             DeprecationWarning,
#         )

#     def add_observation(self, observation, item=None, weight=1.0, validate_eum=True):
#         warnings.warn(
#             "ModelResult.add_observation is deprecated, use Connector instead",
#             DeprecationWarning,
#         )

#     def _in_domain(self, x, y) -> bool:
#         ok = True
#         if self.is_dfsu:
#             ok = self.dfs.contains([x, y])
#         return ok

#     def _validate_observation(self, observation) -> bool:
#         ok = False
#         if self.is_dfsu:
#             if isinstance(observation, PointObservation):
#                 ok = self.dfs.contains([observation.x, observation.y])
#                 if not ok:
#                     raise ValueError("Observation outside domain")
#             elif isinstance(observation, TrackObservation):
#                 ok = True
#         elif self.is_dfs0:
#             # TODO: add check on name
#             ok = True
#         if ok:
#             ok = self._validate_start_end(observation)
#             if not ok:
#                 warnings.warn("No time overlap between model result and observation!")
#         return ok

#     def _validate_start_end(self, observation: Observation) -> bool:
#         try:
#             # need to put this in try-catch due to error in dfs0 in mikeio
#             if observation.end_time < self.dfs.start_time:
#                 return False
#             if observation.start_time > self.dfs.end_time:
#                 return False
#         except:
#             pass
#         return True

#     def _validate_item_eum(
#         self, observation: Observation, item, mod_items=None
#     ) -> bool:
#         """Check that observation and model item eum match"""
#         if mod_items is None:
#             mod_items = self.dfs.items
#         ok = True
#         obs_item = observation.itemInfo
#         if obs_item.type == eum.EUMType.Undefined:
#             warnings.warn(f"{observation.name}: Cannot validate as type is Undefined.")
#             return ok

#         item = self._get_model_item(item, mod_items)
#         if item.type != obs_item.type:
#             ok = False
#             warnings.warn(
#                 f"{observation.name}: Item type should match. Model item: {item.type.display_name}, obs item: {obs_item.type.display_name}"
#             )
#         if item.unit != obs_item.unit:
#             ok = False
#             warnings.warn(
#                 f"{observation.name}: Unit should match. Model unit: {item.unit.display_name}, obs unit: {obs_item.unit.display_name}"
#             )
#         return ok

#     def _get_model_item(self, item, mod_items=None) -> eum.ItemInfo:
#         """Given str or int find corresponding model itemInfo"""
#         if mod_items is None:
#             mod_items = self.dfs.items
#         n_items = len(mod_items)
#         if isinstance(item, int):
#             if (item < 0) or (item >= n_items):
#                 raise ValueError(f"item number must be between 0 and {n_items}")
#         elif isinstance(item, str):
#             item_names = [i.name for i in mod_items]
#             try:
#                 item = item_names.index(item)
#             except ValueError:
#                 raise ValueError(f"item not found in model items ({item_names})")
#         else:
#             raise ValueError("item must be an integer or a string")
#         return mod_items[item]

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

#     def extract(self) -> ComparerCollection:
#         warnings.warn(
#             "ModelResult.extract is deprecated, use Connector instead",
#             DeprecationWarning,
#         )

#     def extract_observation(
#         self,
#         observation: Union[PointObservation, TrackObservation],
#         item: Union[int, str] = None,
#         validate: bool = True,
#     ) -> BaseComparer:
#         """Compare this ModelResult with an observation

#         Parameters
#         ----------
#         observation : <PointObservation> or <TrackObservation>
#             Observation to be compared
#         item : str, integer
#             ModelResult item name or number
#             If None, then try to infer from observation eum value.
#             Default: None
#         validate: bool, optional
#             Validate if observation is inside domain and that eum type
#             and units; Defaut: True

#         Returns
#         -------
#         <fmskill.BaseComparer>
#             A comparer object for further analysis or plotting
#         """
#         if item is None:
#             item = self.item
#         if item is None:
#             item = self._infer_model_item(observation)

#         if validate:
#             ok = self._validate_observation(observation)
#             if ok:
#                 ok = self._validate_item_eum(observation, item)
#             if not ok:
#                 raise ValueError("Could not extract observation")

#         if isinstance(observation, PointObservation):
#             df_model = self._extract_point(observation, item)
#             comparer = PointComparer(observation, df_model)
#         elif isinstance(observation, TrackObservation):
#             df_model = self._extract_track(observation, item)
#             comparer = TrackComparer(observation, df_model)
#         else:
#             raise ValueError("Only point and track observation are supported!")

#         if len(comparer.df) == 0:
#             warnings.warn(f"No overlapping data in found for obs '{observation.name}'!")
#             comparer = None

#         return comparer

#     def _extract_point(self, observation: PointObservation, item) -> pd.DataFrame:
#         ds_model = None
#         if self.is_dfsu:
#             ds_model = self._extract_point_dfsu(observation.x, observation.y, item)
#         elif self.is_dfs0:
#             ds_model = self._extract_point_dfs0(item)

#         return ds_model.to_dataframe()

#     def _extract_point_dfsu(self, x, y, item) -> Dataset:
#         xy = np.atleast_2d([x, y])
#         elemids, _ = self.dfs.get_2d_interpolant(xy, n_nearest=1)
#         ds_model = self.dfs.read(elements=elemids, items=[item])
#         ds_model.items[0].name = self.name
#         return ds_model

#     def _extract_point_dfs0(self, item=None) -> Dataset:
#         if item is None:
#             item = self.item
#         ds_model = self.dfs.read(items=[item])
#         ds_model.items[0].name = self.name
#         return ds_model

#     def _extract_track(self, observation: TrackObservation, item) -> pd.DataFrame:
#         df = None
#         if self.is_dfsu:
#             ds_model = self._extract_track_dfsu(observation, item)
#             df = ds_model.to_dataframe().dropna()
#         elif self.is_dfs0:
#             ds_model = self.dfs.read(items=[0, 1, item])
#             ds_model.items[-1].name = self.name
#             df = ds_model.to_dataframe().dropna()
#             df.index = make_unique_index(df.index, offset_in_seconds=0.01)
#         return df

#     def _extract_track_dfsu(self, observation: TrackObservation, item) -> Dataset:
#         ds_model = self.dfs.extract_track(track=observation.df, items=[item])
#         ds_model.items[-1].name = self.name
#         return ds_model
