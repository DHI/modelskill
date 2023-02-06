from typing import Optional, Union
import warnings

from fmskill import types
from fmskill.data_container import DataContainer


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

        if not isinstance(observations, list):
            observations = [observations]
        if not isinstance(results, list):
            results = [results]

        self._add_data(observations + results)

        print("hold")

    def _add_data(self, data: list[Union[ModelResult, Observation, "Comparer"]]):
        for d in data:
            if isinstance(d, DataContainer):
                if d.is_observation:
                    self.observations.append(d)
                elif d.is_result:
                    self.results.append(d)
            elif isinstance(d, Comparer):
                self._add_data(d.observations + d.results)
            else:
                raise ValueError(f"Unknown data type: {type(d)}")

        _comparison_idc: list[
            tuple[ModelResultIndex, ObservationIndex]
        ] = DataContainer.check_compatibility(self.results + self.observations)

        # Tuples of valid comparisons, format: (result_index, observation_index)
        self._pair_idc = [(m, o - len(self.results)) for m, o in _comparison_idc]

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


if __name__ == "__main__":
    import pandas as pd

    res_1 = ModelResult(
        "tests/testdata/NorthSeaHD_extracted_track.dfs0", item=2, name="North Sea Model"
    )
    res_2 = ModelResult("tests/testdata/Oresund2D.dfsu", item=0, name="Oresund Model")

    obs_1 = Observation(
        pd.read_csv("tests/testdata/altimetry_NorthSea_20171027.csv").set_index("date"),
        item=2,
        name="North Sea Altimetry",
    )
    obs_2 = Observation(
        "tests/testdata/smhi_2095_klagshamn.dfs0",
        x=366844,
        y=6154291,
        item=0,
        name="Klagshamn SMHI",
    )

    c = Comparer([obs_1, obs_2], [res_1, res_2])
