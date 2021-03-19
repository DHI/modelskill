# fmskill
Compare results from [MIKE FM](https://www.mikepoweredbydhi.com/products/mike-21-3) simulations with observations. 

## Purpose

[fmskill](https://github.com/DHI/fmskill) is a python package for scoring MIKE FM models

* Compare different model runs (calibration) or different models (validation)
* Exploratory data analysis - interactive plotting with [plotly](https://plotly.com/python/)
* Publication-ready figures with [matplotlib](https://matplotlib.org/)
* Balance between general and specific needs

Read more about the [vision and scope](vision.md).


## Use cases

[fmskill](https://github.com/DHI/fmskill) wants to be your companion during the different phases of a MIKE FM modelling workflow.

* Model setup - exploratory phase   
* Model calibration
* Model validation and reporting - communicate your final results

## Installation

    > pip install https://github.com/DHI/fmskill/archive/master.zip

## Example notebooks

See examples of use in these notebooks

* [basic.ipynb](https://nbviewer.jupyter.org/github/DHI/fmskill/blob/main/notebooks/basic.ipynb)
* [skill.ipynb](https://nbviewer.jupyter.org/github/DHI/fmskill/blob/main/notebooks/skill.ipynb)
* [Timeseries_compare.ipynb](https://nbviewer.jupyter.org/github/DHI/fmskill/blob/main/notebooks/timeseries_compare.ipynb)
* [Track_comparison.ipynb](https://nbviewer.jupyter.org/github/DHI/fmskill/blob/main/notebooks/Track_comparison.ipynb)
* [SW_DutchCoast.ipynb](https://nbviewer.jupyter.org/github/DHI/fmskill/blob/main/notebooks/SW_DutchCoast.ipynb)
* [Multi_model_comparison.ipynb](https://nbviewer.jupyter.org/github/DHI/fmskill/blob/main/notebooks/Multi_model_comparison.ipynb)


## Design principles

[fmskill](https://github.com/DHI/fmskill) is an object-oriented package built around a few basic concepts:

* ModelResult: defined by a MIKE FM output (.dfsu or .dfs0 file), observations can be added to a ModelResult 
* Observation: e.g. point or track observation
* Metric: can measure the "distance" between a model result and an observation (e.g. bias and rmse)
* Comparer: contains observations and model data interpolated to observation positions and times, can plot and show statistics


## Workflow

1. Define ModelResults
2. Define Observations
3. Associate observations with ModelResults
4. Compare (extract ModelResults at observation positions)
5. Do plotting, statistics, reporting using the ComparerCollection



## Usage

```python
>>> from fmskill.model import ModelResult
>>> from fmskill.observation import PointObservation
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

## Automated reporting

With a few lines of code, it will be possible to generate an automated report.

```python
from fmskill.report import Reporter

rep = Reporter(mr)
rep.markdown()
```

[Very basic first example report](notebooks/HKZN_local/HKZN_local.md)