# Overview

ModelSkill compares model results with observations. The workflow can be split in two phases:

1. **Matching** - making sure that observations and model results are in the same space and time
2. **Analysis** - plots and statistics of the matched data

If the observations and model results are already matched (i.e. are stored in the same data source), 
the `from_matched()` function can be used to go directly to the analysis phase. 
If not, the `match()` function can be used to match the observations and model results in space and time.


## Matching

If the observations and model results are not in the same data source (e.g. dfs0 file), 
they will need to be defined and then matched in space and time with the `match()` function. 
In simple cases, observations and model results can be defined directly in the `match()` function: 

```python
import modelskill as ms
cmp = ms.match("obs.dfs0", "model.dfs0", obs_item="obs_WL", mod_item="WL")
```

But in most cases, the observations and model results will need to be defined separately first.


### Define observations

The observations can be defined as either a `PointObservation` or a `TrackObservation` (a moving point). 

```python
o1 = ms.PointObservation("stn1.dfs0", item="obs_WL")
o2 = ms.PointObservation("stn2.dfs0", item="obs_WL")
```

The `item` needs to be specified as either the item number or the item name if the input file contains multiple items. Several other parameters can be specified, such as the name of the observation, the x- and y-position, and the quantity type and unit of the observation. 


### Define model results

A model result will either be a simple point/track like the observations, or spatial field (e.g. 2d dfsu file) from which the model results will be *extracted* at the observation positions. The following types are available:

* `PointModelResult` - a point result from a dfs0/nc file or a DataFrame
* `TrackModelResult` - a track result from a dfs0/nc file or a DataFrame
* `GridModelResult` - a spatial field from a dfs2/nc file or a Xarray Dataset
* `DfsuModelResult` - a spatial field from a dfsu file

```python
mr1 = ms.PointModelResult("model.dfs0", item="WL_stn1")
mr2 = ms.PointModelResult("model.dfs0", item="WL_stn2")
```

### Match observations and model results

The `match()` function will interpolate the model results to the time (and space) of the observations and return a collection of `Comparer` objects that can be used for analysis. 

```python
cc1 = ms.match(o1, mr1)
cc2 = ms.match(o2, mr2)
cc = cc1 + cc2
```


## Analysis

Once the observations and model results are matched, the `Comparer` object can be used for analysis and plotting. 

