import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from enum import Enum
from copy import deepcopy

from mikeio import Dfs0, Dfsu
from mikefm_skill.observation import PointObservation
import mikefm_skill.metrics as mtr


class ModelResultType(Enum):
    dfs0 = 0
    dfsu = 1
    dfs2 = 2

class ModelResult:
    name = None
    type = None
    filename = None
    dfs = None
    observations = None
    items = None

    def __init__(self, filename, name=None):
        # TODO: add "start" as user may wish to disregard start from comparison
        self.filename = filename
        ext = os.path.splitext(filename)[-1]
        if ext == '.dfsu':
            self.dfs = Dfsu(filename)
            self.type = ModelResultType.dfsu
        #elif ext == '.dfs2':
        #    self.dfs = Dfs2(filename)
        #    self.type = ModelResultType.dfs2
        elif ext == '.dfs0':
            self.dfs = Dfs0(filename)
            self.type = ModelResultType.dfs0
        else:
            raise ValueError(f"Filename extension {ext} not supported (dfsu, dfs0)")

        self.observations = []
        self.items = []

        if name is None:
            name = os.path.basename(filename).split(".")[0]
        self.name = name

    def __repr__(self):
        #return self.dfs
        out = []
        out.append("<mikefm_skill.ModelResult>")
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
        ok = False
        if self.type == ModelResultType.dfsu:
            ok = self.dfs.contains([observation.x, observation.y])
        elif self.type == ModelResultType.dfs0:
            # TODO: add check on name
            ok = True
        
        return ok

    def extract(self):
        """extract model result in observation positions
        """
        res = ModelResultCollection()
        for obs, item in zip(self.observations, self.items):
            mrp = self._extract_point_observation(obs, item)
            res.add_result(mrp)
        return res

    def _extract_point_observation(self, observation, item):
        ds_model = None
        if self.type == ModelResultType.dfsu:
            ds_model = self._extract_point_dfsu(observation, item)
        elif self.type == ModelResultType.dfs0:
            ds_model = self._extract_point_dfs0(observation, item)

        return ModelResultPoint(observation, ds_model)

    def _extract_point_dfsu(self, observation, item):
        xy = np.atleast_2d([observation.x, observation.y])
        elemids, _ = self.dfs.get_2d_interpolant(xy, n_nearest=1)
        ds_model = self.dfs.read(elements=elemids, items=[item])
        ds_model.items[0].name = self.name
        return ds_model

    def _extract_point_dfs0(self, observation, item):
        ds_model = self.dfs.read(items=[item])
        ds_model.items[0].name = self.name
        return ds_model

    def plot_observation_positions(self, figsize=None):
        if self.type == ModelResultType.dfs0:
            warnings.warn("Plotting observations is only supported for dfsu ModelResults")
            return
        xn = self.dfs.node_coordinates[:,0]
        offset_x = 0.02*(max(xn) - min(xn))
        ax = self.dfs.plot(plot_type='outline_only', figsize=figsize)
        for obs in self.observations:
            ax.scatter(x=obs.x, y=obs.y, marker='x')
            ax.annotate(obs.name, (obs.x + offset_x, obs.y))


# TODO: find better name: PointComparer
# TODO: add more ModelResults
class ModelResultPoint:
    observation = None
    mod_name = None
    obs_name = "Observation"
    mod_df = None
    df = None
    #stats = None

    @property
    def name(self):
        return self.observation.name

    @property
    def residual(self):
        return self.mod - self.obs
    
    @property
    def obs(self):
        return self.df[self.obs_name].values 

    @property
    def mod(self):
        return self.df[self.mod_name].values

    def __init__(self, observation, modeldata):
        self.observation = deepcopy(observation)
        self.mod_df = modeldata.to_dataframe()
        self.mod_name = self.mod_df.columns[0]
        self.observation.subset_time(start=modeldata.start_time, end=modeldata.end_time)
        self.df = self._model2obs_interp(self.observation, modeldata)

    def _model2obs_interp(self, obs, mod_ds):
        """interpolate model to measurement time
        """        
        df = mod_ds.interp_time(obs.time).to_dataframe()
        df[self.obs_name] = obs.values
        return df


    def remove_bias(self, correct='Model'):
        bias = self.residual.mean()
        if correct == 'Model':
            self.mod_df[self.mod_name] = self.mod_df.values - bias
            self.df[self.mod_name] = self.df[self.mod_name].values - bias
        elif correct == 'Observation':
            self.df[self.obs_name] = self.df[self.obs_name].values + bias
        else:
            raise ValueError(f"Unknown correct={correct}. Only know 'Model' and 'Observation'")


    def plot_timeseries(self, figsize=None):
        fig, ax = plt.subplots(1,1,figsize=figsize)
        self.mod_df.plot(ax=ax)
        self.df[self.obs_name].plot(marker='.', linestyle = 'None', ax=ax)
        ax.legend([self.mod_name, self.obs_name]);

    def scatter(self, xlabel=None, ylabel=None, binsize=None, nbins=100, backend='matplotlib', title="", **kwargs):    

        x = self.df[self.obs_name].values
        y = self.df[self.mod_name].values

        if xlabel is None:
            xlabel = "Observation"

        if ylabel is None:
            ylabel = "Model"

        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        if binsize is None:
            binsize = (xmax - xmin) / nbins
        else:
            nbins = int((xmax - xmin) / binsize)

        xq = np.quantile(x,q=np.linspace(0, 1, num=nbins))
        yq = np.quantile(y,q=np.linspace(0, 1, num=nbins))

        if backend == 'matplotlib':
            plt.plot([xmin,xmax],[ymin,ymax], label='1:1')
            plt.plot(xq, yq,label='QQ',c='gray')
            plt.hist2d(x, y, bins=nbins, cmin=0.01, **kwargs)
            plt.legend()
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.axis('equal')
            cbar = plt.colorbar()
            cbar.set_label('# points')
            plt.title(title)
        
        elif backend == 'bokeh':
            import bokeh.plotting as bh
            import bokeh.models as bhm
            #bh.output_notebook()

            p = bh.figure(x_axis_label='obs', y_axis_label='model', title=title)
            p.hexbin(x, y, size=binsize)
            p.scatter(xq, yq, legend_label="Q-Q", color='gray')
            
            linvals = np.linspace(np.min([x,y]), np.max([x,y]))
            p.line(linvals, linvals, legend_label="1:1")
            #mder = bhm.Label(x=10, y=500, x_units='screen', y_units='screen',
            #                text=f"MdEr: {MdEr(x,y):.2f}")
            #rmse = bhm.Label(x=10, y=480, x_units='screen', y_units='screen',
            #                text=f"RMSE: {RMSE(x,y):.2f}")
            #stder = bhm.Label(x=10, y=460, x_units='screen', y_units='screen',
            #                text=f"StdEr: {StdEr(x,y):.2f}")
            #N = bhm.Label(x=10, y=440, x_units='screen', y_units='screen',
            #                text=f"N: {len(x)}")

            #p.add_layout(mder)
            #p.add_layout(rmse)
            #p.add_layout(stder)
            #p.add_layout(N)
            bh.show(p)

        else:
            raise ValueError(f"Plotting backend: {backend} not supported")


    def residual_hist(self, bins=None):
        return plt.hist(self.residual, bins=bins)

    def statistics(self):
        resi = self.residual
        bias = resi.mean()
        uresi = resi - bias
        rmse = np.sqrt(np.mean(uresi**2))
        return bias, rmse

    def skill(self, metric=mtr.rmse):
        return metric(self.obs, self.mod)

class ModelResultCollection:

    results = []

    def __getitem__(self, x):
        return self.results[x]


    def add_result(self, modelresult):

        self.results.append(modelresult)

    def skill(self, metric=mtr.rmse):

        scores = [metric(mr.obs, mr.mod) for mr in self.results]

        return np.average(scores)

    def skill_report(self, metrics:list = None) -> pd.DataFrame:

        if metrics is None:
            metrics = [mtr.bias, mtr.rmse, mtr.corr_coef, mtr.scatter_index ]

        res = {}
        for mr in self.results:
            tmp = {}
            for metric in metrics:
                tmp[metric.__name__] =  metric(mr.obs, mr.mod)
            res[mr.name] = tmp
        
        return pd.DataFrame(res).T
             


        
