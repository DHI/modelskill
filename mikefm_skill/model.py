import os
import numpy as np
import warnings
from mikeio import Dfs0, Dfsu
from mikefm_skill.observation import PointObservation

class ModelResult:
    name = None
    filename = None
    dfs = None
    observations = []
    items = []

    def __init__(self, filename, name=None):
        # TODO: add "start" as user may wish to disregard start from comparison
        self.filename = filename
        self.dfs = Dfsu(filename)

        if name is None:
            name = os.path.basename(filename).split(".")[0]
        self.name = name

    def __repr__(self):
        #return self.dfs
        out = []
        out.append("ModelResult")
        out.append(self.filename)
        return str.join("\n", out)

    def add_observation(self, observation, item):
        ok = self._validate_observation(observation)
        if ok:
            self.observations.append(observation)
            self.items.append(item)
        else:
            warnings.warn('Could not add observation')

    def _validate_observation(self, observation):
        return self.dfs.contains([observation.x, observation.y])

    def extract(self):
        """extract model result in observation positions
        """
        ex = []
        for j, obs in enumerate(self.observations):
            mrp = self._extract_point_observation(obs, self.items[j])
            ex.append(mrp)
        return ex

    def _extract_point_observation(self, observation, item):
        elemids, _ = self.dfs.get_2d_interpolant([observation.x, observation.y], n_nearest=1)
        ds_model = self.dfs.read(elements=elemids, items=[item])
        return ModelResultPoint(observation, ds_model)

    def plot_observation_positions(self):
        ax = self.dfs.plot(plot_type='outline_only')
        for obs in enumerate(self.observations):
                
            ax.plot()
        #pass


# TODO: find better name
class ModelResultPoint:
    #observation = None
    ds = None
    mod_ds = None
    df = None
   
    def __init__(self, observation, modeldata):
        self.observation = observation
        self.mod_ds = modeldata
        observation.subset_time(start=modeldata.start_time, end=modeldata.end_time)
        self.df = self._model2obs_interp(observation.ds, modeldata)


    def _model2obs_interp(self, obs_ds, mod_ds):
        """interpolate model to measurement time
        """        
        df = mod_ds.interp_time(obs_ds.time).to_dataframe()
        df['observation'] = obs_ds.data[0]


    def plot_timeseries(self):
        pass

    def scatter(self):
        pass

    def statistics(self):
        pass