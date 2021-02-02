import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
#from copy import deepcopy
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
        ds_model.items[0].name = self.name
        return ModelResultPoint(observation, ds_model)

    def plot_observation_positions(self, figsize=None):
        ax = self.dfs.plot(plot_type='outline_only', figsize=figsize)
        for obs in self.observations:
            ax.scatter(x=obs.x, y=obs.y, marker='x')
            ax.annotate(obs.name, (obs.x, obs.y))


# TODO: find better name
class ModelResultPoint:
    observation = None
    mod_name = None
    obs_name = "Observation"
    mod_df = None
    df = None
    stats = None

    @property
    def residual(self):
        obs = self.df[self.obs_name].values 
        mod = self.df[self.mod_name].values
        return mod - obs

    def __init__(self, observation, modeldata):
        self.observation = observation #deepcopy(observation)
        self.mod_df = modeldata.to_dataframe()
        self.mod_name = self.mod_df.columns[0]
        self.observation.subset_time(start=modeldata.start_time, end=modeldata.end_time)
        self.df = self._model2obs_interp(self.observation.ds, modeldata)

    def _model2obs_interp(self, obs_ds, mod_ds):
        """interpolate model to measurement time
        """        
        df = mod_ds.interp_time(obs_ds.time).to_dataframe()
        df[self.obs_name] = obs_ds.data[0]
        return df

    def plot_timeseries(self, figsize=None):
        fig, ax = plt.subplots(1,1,figsize=figsize)
        self.mod_df.plot(ax=ax)
        self.df[self.obs_name].plot(marker='.', linestyle = 'None', ax=ax)
        ax.legend([self.mod_name, self.obs_name]);

    def scatter(self):
        pass

    def statistics(self):
        resi = self.residual
        bias = resi.mean()
        #rmse = 
        pass