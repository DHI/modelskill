# mikefm-skill
Compare results from [MIKE FM](https://www.mikepoweredbydhi.com/products/mike-21-3) simulations with observations

## Purpose/objective/scope

[mikefm-skill](https://github.com/DHI/mikefm-skill) is a python package 

* Score MIKE FM models
    - Provide a wide range of common evaluation metrics: 
    - Single-number evaluation (aggregation) - e.g. cost function for calibration
    - Spatial aggregation 
* Compare different model runs (calibration) or different models (validation)
* Exploratory data analysis - interactive plotting with [plotly](https://plotly.com/python/)
* Report/publication-ready figures with [matplotlib](https://matplotlib.org/)
* Balance between general and specific needs
    - Primarily support dfs files (using [mikeio](https://github.com/DHI/mikeio))
    - Handle circular variables such as wave direction
    - Handle vector variables such as u- and v-components of current
    - Tidal analysis

## Use cases

[mikefm-skill](https://github.com/DHI/mikefm-skill) wants to be your companion during the different phases of a MIKE FM modelling workflow.

* Model setup - exploratory phase
    - Explore timeseries, histogram and scatter plots of model and observation
    - Assess quality of observations (find outliers)     
* Model calibration
    - x
* Model validation and reporting - communicate your final results
    - Prepare figures for report or html 

## Installation

    > pip install https://github.com/DHI/mikefm-skill/archive/master.zip

## Example notebooks

See examples of use in these notebooks

* [basic.ipynb](https://nbviewer.jupyter.org/github/DHI/mikefm-skill/blob/main/notebooks/basic.ipynb)
* [skill.ipynb](https://nbviewer.jupyter.org/github/DHI/mikefm-skill/blob/main/notebooks/skill.ipynb)
* [timeseries_compare.ipynb](https://nbviewer.jupyter.org/github/DHI/mikefm-skill/blob/main/notebooks/timeseries_compare.ipynb)
* [SW_DutchCoast.ipynb](https://nbviewer.jupyter.org/github/DHI/mikefm-skill/blob/main/notebooks/SW_DutchCoast.ipynb)
* [Multi_model_comparison.ipynb](https://nbviewer.jupyter.org/github/DHI/mikefm-skill/blob/main/notebooks/Multi_model_comparison.ipynb)


## Design (principles)

[mikefm-skill](https://github.com/DHI/mikefm-skill) is build around a few basic concepts:

* Observation: 
* ModelResult: defined by a MIKE FM output (.dfsu or .dfs0 file), observations can be added to a ModelResult 
* Metric: can measure the "distance" between a model result and an observation (e.g. bias and rmse)
* Comparer: 
* ComparisonCollection: 


## Usage
```python
>>> from mikefm_skill.model import ModelResult
>>> from mikefm_skill.observation import PointObservation
>>> mr = ModelResult("Oresund2D.dfsu")
>>> klagshamn = PointObservation("smhi_2095_klagshamn.dfs0", item=0, x=366844, y=6154291, name="Klagshamn")
>>> drogden = PointObservation("dmi_30357_Drogden_Fyr.dfs0", item=0, x=355568, y=6156863)
>>> mr.add_observation(klagshamn, item=0)
>>> mr.add_observation(drogden, item=0)
>>> collection = mr.extract()
>>> collection.skill_report()
                       bias  rmse  corr_coef  scatter_index
Klagshamn              0.18  0.19       0.84           0.32
dmi_30357_Drogden_Fyr  0.26  0.28       0.51           0.53
```

### Overview of observation locations

![map](images/map.png)

### Scatter plot

![scatter](images/scatter.png)

### Timeseries plot

Timeseries plots can either be static and report-friendly ([matplotlib](https://matplotlib.org/)) or interactive with zoom functionality ([plotly](https://plotly.com/python/)).

![timeseries](images/plotly_timeseries.png)
