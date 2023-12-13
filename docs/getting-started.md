Getting started
===============

This page describes the typical ModelSkill workflow for comparing model
results and observations. If you just need a simple one-to-one time
series comparison, see the [simple time series
comparison](simple-compare.md).

Workflow
--------

The typical ModelSkill workflow consists of these five steps:

1.  Define **ModelResults**
2.  Define **Observations**
3.  **Connect** observations and ModelResults
4.  **Extract** ModelResults at observation positions
5.  Do analysis, plotting, etc with a **Comparer**

### 1. Define ModelResults

The result of a MIKE 21/3 simulation is stored in one or more dfs files.
The most common formats are .dfsu for distributed data and .dfs0 for
time series point data. A ModelSkill
[ModelResult](api/model.md#modelskill.model.PointModelResult) is defined by the
result file path and a name:

```python
import modelskill as ms
mr = ms.ModelResult("SW/HKZN_local_2017_DutchCoast.dfsu", name='HKZN_local', item="Sign. Wave Height")
```

Currently, ModelResult supports .dfs0 and .dfsu files and pandas
DataFrame. Only the file header is read when the ModelResult object is
created. The data will be read later.

### 2. Define Observations

The next step is to define the measurements to be used for the skill
assessment. Two types of observation are available:

-   [PointObservation](api/observation.md#modelskill.observation.PointObservation)
-   [TrackObservation](api/observation.md#modelskill.observation.TrackObservation)

Let\'s assume that we have one PointObservation and one
TrackObservation:

```python
hkna = ms.PointObservation("HKNA_Hm0.dfs0", item=0, x=4.2420, y=52.6887, name="HKNA")
c2 = ms.TrackObservation("Alti_c2_Dutch.dfs0", item=3, name="c2")
```

In this case both observations are provided as .dfs0 files but pandas
dataframes are also supported in case data are stored in another file
format.

Both PointObservation and TrackObservation need the path of the data
file, the item number (or item name) and a name. A PointObservation
further needs to be initialized with it\'s x-, y-position.

### 3. Connect observations and ModelResults

```python
cc = ms.match([hkna, c2], mr)
```

This method returns a
[ComparerCollection](api/compare.md#modelskill.comparison.ComparerCollection)
for further analysis and plotting.

### 4. Do analysis, plotting, etc with a Comparer

The object returned by the `match()` method is a *Comparer*/*ComparerCollection*. It holds the matched observation and model data and has methods for plotting and
skill assessment.

The primary comparer methods are:

- [skill()](api/compare.md#modelskill.comparison.ComparerCollection.skill)
  which returns a table with the skill scores
- various plot methods of the comparer objects
    * `plot.scatter()`
    * `plot.timeseries()`
    * `plot.kde()`
    * `plot.qq()`
    * `plot.hist()`

### 5. Save / load the ComparerCollection

It can be useful to save the comparer collection for later use. This can be done using the `save()` method:

```python
cc.save("my_comparer_collection.msk")
```

The comparer collection can be loaded again from disk, using the `load()` method:

```python
cc = ms.load("my_comparer_collection.msk")
```


#### Filtering

In order to select only a subset of the data for analysis, the comparer has a `sel()` method which returns a new comparer with the selected data. 

This method allow filtering of the data in several ways:

-   on `observation` by specifying name or id of one or more
    observations
-   on `model` (if more than one is compared) by giving name or id
-   temporal using the `start` and `end` arguments
-   spatial using the `area` argument given as a bounding box or a
    polygon
