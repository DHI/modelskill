# Observations

ModelSkill supports two types of observations:
 
* [`PointObservation`](point.md) - a point timeseries from a dfs0/nc file or a DataFrame
* [`TrackObservation`](track.md) - a track (moving point) timeseries from a dfs0/nc file or a DataFrame

An observation can be created by explicitly invoking one of the above classes or using the [`observation()`](observation.md) function which will return the appropriate type based on the input data (if possible).

