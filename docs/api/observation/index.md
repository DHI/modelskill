# Observations

ModelSkill supports two types of observations:
 
* `PointObservation` - a point timeseries from a dfs0/nc file or a DataFrame
* `TrackObservation` - a track (moving point) timeseries from a dfs0/nc file or a DataFrame

An observation can be created by explicitly invoking one of the above classes or using the `observation()` function which will return the appropriate type based on the input data (if possible).


## observation()

::: modelskill.observation
