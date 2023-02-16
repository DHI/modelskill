import warnings
from typing import Optional, Union, Iterable

import xarray as xr

from fmskill import types, parsing
from fmskill.data_container import DataContainer, compare
from fmskill.settings import options, reset_option


class Observation:
    def __new__(
        self,
        data: types.DataInputType,
        item: types.ItemSpecifier = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        name: Optional[str] = None,
    ) -> DataContainer:

        return DataContainer(
            data=data,
            item=item,
            is_observation=True,
            x=x,
            y=y,
            name=name,
        )


class ModelResult:
    def __new__(
        self,
        data: types.DataInputType,
        item: types.ItemSpecifier = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        name: Optional[str] = None,
    ) -> DataContainer:

        return DataContainer(
            data=data,
            item=item,
            is_result=True,
            x=x,
            y=y,
            name=name,
        )


ObservationIndex = int
ModelResultIndex = int


class Comparer:
    def __init__(
        self,
        observations: Union[Observation, list[Observation]],
        results: Union[ModelResult, list[ModelResult]],
    ) -> None:

        self.observations = []
        self.results = []
        self.extracted = {}

        self.observation_names = set()
        self.result_names = set()
        self.unextracted_observation_names = set()
        self.unextracted_result_names = set()

        if not isinstance(observations, list):
            observations = [observations]
        if not isinstance(results, list):
            results = [results]

        self._add_data(observations + results)

    @property
    def metrics(self):
        return options.metrics.list

    @metrics.setter
    def metrics(self, values) -> None:
        if values is None:
            reset_option("metrics.list")
        else:
            options.metrics.list = parsing.parse_metric(values)

    @property
    def metric_names(self):
        return [m.__name__ for m in self.metrics]

    @property
    def extracted_result_names(self):
        return self.result_names - self.unextracted_result_names

    @property
    def extracted_observation_names(self):
        return self.observation_names - self.unextracted_observation_names

    def __add__(self, other: Union[ModelResult, Observation, "Comparer"]):
        if not isinstance(other, list):
            other = [other]
        assert all(
            isinstance(i, (Comparer, DataContainer)) for i in other
        ), "Can only add Comparer, ModelResult or Observation"

        self._add_data(other)
        return self

    def __getitem__(self, key):
        if not self.extracted:
            warnings.warn("No data extracted. Use Comparer.extract() first.")
            return

        if isinstance(key, str):
            if key in self.extracted:
                return self.extracted[key]

            else:
                partial_matches = [
                    k for k in self.extracted.keys() if key.lower() in k.lower()
                ]
                if len(partial_matches) > 1:
                    return {k: self.extracted[k] for k in partial_matches}
                elif len(partial_matches) == 1:
                    return self.extracted[partial_matches[0]]

        elif isinstance(key, int):
            if key < len(self.extracted):
                return list(self.extracted.values())[key]
            elif -len(self.extracted) < key < 0:
                return list(self.extracted.values())[key + len(self.extracted)]
            else:
                raise IndexError(f"Index out of range: {key}")

    def _add_data(self, data: list[Union[DataContainer, "Comparer"]]):
        for d in data:
            if isinstance(d, DataContainer):
                if d.is_observation:
                    if d.name in self.observation_names:
                        warnings.warn(
                            f"Duplicate observation name: {d.name}. Please choose unique names."
                        )
                    else:
                        self.observations.append(d)
                        self.observation_names.add(d.name)
                        self.unextracted_observation_names.add(d.name)

                elif d.is_result:
                    if d.name in self.result_names:
                        warnings.warn(
                            f"Duplicate result name: {d.name}. Please choose unique names."
                        )
                    else:
                        self.results.append(d)
                        self.result_names.add(d.name)
                        self.unextracted_result_names.add(d.name)
            elif isinstance(d, Comparer):
                self._add_data(d.observations + d.results)
            else:
                raise ValueError(f"Unknown data type: {type(d)}")

    def _add_skill_metrics(self):

        for ds in self.extracted.values():

            obs_values = ds.variable.sel(source="Observation").values

            def _get_obs_metric(x, metric):
                np_metric = metric(obs_values, x.variable.values)
                return xr.DataArray(np_metric)

            for m in self.metrics:
                ds[m.__name__] = ds.groupby("source").map(_get_obs_metric, metric=m)

    def extract(self):
        _results_to_add = [
            r for r in self.results if r.name in self.unextracted_result_names
        ]
        _observations_to_add = [
            o for o in self.observations if o.name in self.unextracted_observation_names
        ]
        if _observations_to_add and not _results_to_add:
            _results_to_add = self.results
        else:
            _observations_to_add = self.observations
            _results_to_add = self.results
        new_data = compare(_results_to_add + _observations_to_add)
        self.extracted.update(new_data)

        self._add_skill_metrics()
        self._set_skill()

        self.unextracted_result_names.clear()
        self.unextracted_observation_names.clear()

    def plot_observation_positions(self, title=None, figsize=None):
        from fmskill.plot import plot_observation_positions

        res_idc_with_geom = [
            i for i, r in enumerate(self.results) if r.geometry is not None
        ]
        if not res_idc_with_geom:
            warnings.warn("Only supported for dfsu ModelResults")
            return

        return plot_observation_positions(
            self.results[res_idc_with_geom[0]].geometry,
            self.observations,
            title=title,
            figsize=figsize,
        )

    def plot_temporal_coverage(
        self,
        *,
        show_model=True,
        limit_to_model_period=True,
        marker="_",
        title=None,
        figsize=None,
    ):

        from fmskill.plot import plot_temporal_coverage

        return plot_temporal_coverage(
            modelresults=self.results if show_model else [],
            observations=self.observations,
            limit_to_model_period=limit_to_model_period,
            marker=marker,
            title=title,
            figsize=figsize,
        )

    def _set_skill(self):
        _relevant_arrays = [v[self.metric_names] for v in self.extracted.values()]
        ds = xr.concat(_relevant_arrays, dim="observation").sel(
            source=list(self.result_names)
        )
        ds = ds.assign_coords(observation=list(self.extracted.keys()))
        ds["observation"] = ds.observation.astype("object")
        ds = ds.rename({"source": "model"})
        self._skill = ds

    def skill(self, observation=None, **kwargs) -> xr.Dataset:
        obs = self._get_obs(observation)
        return self._skill.sel(observation=obs, **kwargs)

    def _get_obs(self, obs):
        if obs is None:
            return list(self.extracted_observation_names)
        if isinstance(obs, str):
            if obs not in self.observation_names:
                raise ValueError(f"Observation not found: {obs}")
            return [obs]
        if isinstance(obs, Iterable):
            if not set(obs).issubset(self.observation_names):
                warnings.warn(
                    f"Observations not found: {set(obs).difference(self.observation_names)}"
                )
            return list(set(obs).intersection(self.observation_names))


if __name__ == "__main__":
    fldr = "tests/testdata/SW/"
    o1 = Observation(fldr + "HKNA_Hm0.dfs0", item=0, x=4.2420, y=52.6887, name="HKNA")
    o2 = Observation(fldr + "eur_Hm0.dfs0", item=0, x=3.2760, y=51.9990, name="EPL")
    o3 = Observation(fldr + "Alti_c2_Dutch.dfs0", item=3, name="c2")

    mr1 = ModelResult(fldr + "HKZN_local_2017_DutchCoast.dfsu", name="SW_1", item=0)
    mr2 = ModelResult(fldr + "HKZN_local_2017_DutchCoast_v2.dfsu", name="SW_2", item=0)

    c = Comparer([o1, o2, o3], [mr1, mr2])

    c.extract()

    print("hold")
