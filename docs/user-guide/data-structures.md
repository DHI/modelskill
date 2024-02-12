# Data Structures

The main data structures in ModelSkill can be grouped into three categories:

* **Primary** data (observations and model results)
* **Comparer** objects
* **Skill** objects

All objects share some common principles:

* The data container is accesssible via the `data` attribute. 
* The data container is an `xarray` object (except for the `SkillTable` object, which is a `pandas` object).
* The main data selection method is `sel`, which is a wrapper around `xarray.Dataset.sel`.
* All plotting are accessible via the `plot` accessor of the object.


## Observations and model results

The primary data of ModelSkill are the data that needs to be compared: observations and model results. The underlying data structures are very similar and can be grouped according to the spatial dimensionality (`gtype`) of the data:

* `point`: 0D time series data
* `track`: 0D time series data at moving locations (trajectories)
* `grid`: gridded 2D data
* `dfsu`: flexible mesh 2D data

Point and track data are both `TimeSeries` objects, while grid and dfsu data are both `SpatialField` objects. `TimeSeries` objects are ready to be compared whereas data from `SpatialField` object needs to be *extracted* first (the extracted object will be of the `TimeSeries` type).

`TimeSeries` objects contains its data in an `xarray.Dataset` with the actual data in the first DataArray and optional auxilliary data in the following DataArrays. The DataArrays have a `kind` attribute with either `observation` or `model`.


## Comparer objects

Comparer objects are results of a matching procedure (between observations and model results) or constructed directly from already matched data. A comparison of a *single* observation and one or more model results are stored in a `Comparer` object. A comparison of *multiple* observations and one or more model results are stored in a `ComparerCollection` object which is a collection of `Comparer` objects.

The matched data in a `Comparer` is stored in an `xarray.Dataset` which can be accessed via the `data` attribute. The Dataset has an attribute `gtype` which is a string describing the type of data (e.g. `point`, `track`). The first DataArray in the Dataset is the observation data, the next DataArrays are model result data and optionally additional DataArrays are auxilliarye data. Each of the DataArrays have a `kind` attribute with either `observation`, `model` or `aux`.

Both `Comparer` and `ComparerCollection` have a `plot` accessor for plotting the data (e.g. `cmp.plot.timeseries()` or `cmp.plot.scatter()`).



## Skill objects

Calling a skill method on a comparer object will return a skill object with skill scores (statistics) from comparing observation and model result data using different metrics (e.g. root mean square error). Two skill objects are currently implemented: `SkillTable` and `SkillGrid`. The first is relevant for all ModelSkill users while the latter is relevant for users of the track data (e.g. MetOcean studies using satellite altimetry data).

If `c` is a comparer object, then the following skill methods are available:

* `c.skill()` -> `SkillTable`
* `c.mean_skill()` -> `SkillTable`
* `c.gridded_skill()` -> `SkillGrid`


### SkillTable

