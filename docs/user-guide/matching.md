# Matching 

Once observations and model results have been defined, the next step is to match them. This is done using the `match()` function which handles the allignment of the observation and model result data in space and time. Note that if the data is already matched, the `from_matched()` function can be used to create a `Comparer` directly from the matched data and the matching described here is not needed.

The observation is considered the *truth* and the model result data is therefore interpolated to the observation data positions.

The matching process will be different depending on the geometry of observation and model result:

* Geometries are the *same* (e.g. both are point time series): only temporal matching is needed
* Geometries are *different* (e.g. observation is a point time series and model result is a grid): data is first spatially *extracted* from the model result and *then* matched in time. 


## Temporal matching

Temporal matching is done by interpolating the model result data to the observation data time points; it is carried out after spatial matching when applicable. The interpolation is *linear* in time and done inside the `match()` function.


## Matching of time series

If observation and model result are of the same geometry, the matching is done *one* observation at a time. Several model results can be matched to the same observation. The result of the matching process is a `Comparer` object which contains the matched data. 

In the most simple cases, one observation to one model result, the `match()` function can be used directly, without creating Observation and ModelResult objects first:

```python
>>> cmp = ms.match('obs.dfs0', 'model.dfs0', obs_item='obs_WL', mod_item='WL')
```

In all other cases, the observations and model results needs to be defined first.

```python
>>> o = ms.observation('obs.dfs0', item='waterlevel')
>>> mr1 = ms.model_result('model1.dfs0', item='WL1')
>>> mr2 = ms.model_result('model2.dfs0', item='WL2')
>>> cmp = ms.match(o, [mr1, mr2])
```

In most cases, *several* observations needs to matched with several model results. This can be done by constructing a list of `Comparer` objects and then combining them into a `ComparerCollection`:

```python
>>> cmps = []
>>> for o in observations:
>>>     mr1 = ...
>>>     mr2 = ...
>>>     cmps.append(ms.match(o, [mr1, mr2]))
>>> cc = ms.ComparerCollection(cmps)
```



## Matching with dfsu or grid model result

If the model result is a SpatialField, i.e., either a `GridModelResult` or a `DfsuModelResult`, and the observation is of lower dimension (e.g. point), then the model result needs to be *extracted* before matching can be done. This can be done "offline" before using ModelSkill, e.g., using [MIKE](https://www.mikepoweredbydhi.com/) tools or [MIKE IO](https://github.com/DHI/mikeio), or as part of the matching process using ModelSkill. We will here focus on the latter. 

In this situation, *multiple* observations can be matched to the same model result, in which case the `match` function returns a `ComparerCollection` instead of a `Comparer` which is the returned object for single observation matching. 

```python
>>> o1 = ms.observation('obs1.dfs0', item='waterlevel')
>>> o2 = ms.observation('obs2.dfs0', item='waterlevel')
>>> mr = ms.model_result('model.dfsu', item='WaterLevel')
>>> cc = ms.match([o1, o2], mr)   # returns a ComparerCollection
```

Matching `PointObservation` with `SpatialField` model results consists of two steps: 

1. Extracting data from the model result at the spatial position of the observation, which returns a PointModelResult
2. Matching the extracted data with the observation data in time

Matching `TrackObservation` with `SpatialField` model results is for technical reasons handled in *one* step, i.e., the data is extracted in both space and time.

The spatial matching method (selection or interpolation) can be specified using the `spatial_method` argument of the `match()` function. The default method depends on the type of observation and model result as specified in the sections below.


### Extracting data from a DfsuModelResult

Extracting data for a specific point position from the flexible mesh dfsu files can be done in several ways (specified by the `spatial_method` argument of the `match()` function): 

* Selection of the "contained" element 
* Selection of the "nearest" element (often the same as the contained element, but not always)
* Interpolation with "inverse_distance" weighting (IDW) using the five nearest elements (default)

The default (inverse_distance) is not necessarily the best method in all cases. When the extracted position is close to the model boundary, "contained" may be a better choice.

```python
>>> cc = ms.match([o1, o2], mr_dfsu, spatial_method='contained')   
```

Note that extraction of *track* data does not currently support the "contained" method.

Note that the extraction of point data from 3D dfsu files is not yet fully supported. It is recommended to extract the data "offline" prior to using ModelSkill.


### Extracting data from a GridModelResult

Extracting data from a GridModelResult is done through xarray's `interp()` function. The `spatial_method` argument of the `match()` function is passed on to the `interp()` function as the `method` argument. The default method is "linear" which is the recommended method for most cases. Close to land where the grid model result data is often missing, "nearest" may be a better choice.

```python
>>> cc = ms.match([o1, o2], mr_netcdf, spatial_method='nearest')   
```


## Event-based matching and handling of gaps

If the model result data contains gaps either because only events are stored or because of missing data, the `max_model_gap` argument of the `match()` function can be used to specify the maximum allowed gap (in seconds) in the model result data. This will avoid interpolating model data over long gaps in the model result data!


## Multiple model results with different temporal coverage

If the model results have different temporal coverage, the `match()` function will only match the overlapping time period to ensure that the model results are comparable. The `Comparer` object will contain the matched data for the overlapping period only.
