# Model Result

## Types of model results

A model result can either be a simple point/track, or spatial field (e.g. 2d dfsu file) from which data can be *extracted* at the observation positions by spatial interpolation. The following types are available:

* Timeseries
    - `PointModelResult` - a point result from a dfs0/nc file or a DataFrame
    - `TrackModelResult` - a track (moving point) result from a dfs0/nc file or a DataFrame
* SpatialField (extractable)
    - `GridModelResult` - a spatial field from a dfs2/nc file or a Xarray Dataset
    - `DfsuModelResult` - a spatial field from a dfsu file

A model result can be created by explicitly invoking one of the above classes or using the `model_result()` function which will return the appropriate type based on the input data (if possible).


## model_result()

::: modelskill.model_result
