# Getting started


This page describes the typical ModelSkill workflow for comparing model
results and observations. 


## Workflow


The typical ModelSkill workflow consists of these four steps:

1.  Define **Observations**
2.  Define **ModelResults**
3.  **Match** observations and ModelResults in space and time
4.  Do analysis, plotting, etc with a **Comparer**



### Define Observations

The first step is to define the measurements to be used for the skill
assessment. Two types of observation are available:

-   [PointObservation](../api/observation/point.md)
-   [TrackObservation](../api/observation/track.md)

Let's assume that we have one PointObservation and one
TrackObservation (`name` is used to identify the observation, similar to the `name` of the model above). 

```python hl_lines="3 5"
hkna = ms.PointObservation("HKNA_Hm0.dfs0", item=0,
                            x=4.2420, y=52.6887,
                            name="HKNA")
c2 = ms.TrackObservation("Alti_c2_Dutch.dfs0", item=3,
                          name="c2")
```

In this case both observations are provided as .dfs0 files but pandas
dataframes are also supported in case data are stored in another file
format.

Both PointObservation and TrackObservation need the path of the data
file, the item number (or item name) and a name. A PointObservation
further needs to be initialized with it\'s x-, y-position.


### Define ModelResults

The result of a simulation is stored in one or more result files, e.g. dfsu, dfs0, nc, csv.

The name is used to identify the model result in the plots and tables.

```python hl_lines="4"
import modelskill as ms
mr = ms.DfsuModelResult("SW/HKZN_local_2017_DutchCoast.dfsu", 
                         item="Sign. Wave Height",
                         name='HKZN_local')
```



### Match observations and ModelResults

This [match()](../api/matching.md/#modelskill.match) method returns a [Comparer](../api/comparer.md#modelskill.Comparer) (a single observation) or a
[ComparerCollection](../api/comparercollection.md#modelskill.ComparerCollection) (multiple observations)
for further analysis and plotting.

```python
cc = ms.match([hkna, c2], mr)
```



### Do analysis, plotting, etc with a Comparer

The object returned by the `match()` method is a *Comparer*/*ComparerCollection*. It holds the matched observation and model data and has methods for plotting and
skill assessment.

The primary comparer methods are:

- [skill()](../api/comparercollection.md#modelskill.ComparerCollection.skill)
  which returns a [SkillTable](../api/skill.md) with the skill scores
- various [plot](../api/comparercollection.md/#modelskill.comparison._collection_plotter.ComparerCollectionPlotter) methods of the comparer objects (e.g. `plot.scatter()`, `plot.timeseries()`)
- [sel()](../api/comparercollection.md/#modelskill.ComparerCollection.sel) method for selecting data
    

### Save / load the ComparerCollection

It can be useful to save the comparer collection for later use. This can be done using the `save()` method:

```python
cc.save("my_comparer_collection.msk")
```

The comparer collection can be loaded again from disk, using the `load()` method:

```python
cc = ms.load("my_comparer_collection.msk")
```


### Filtering

In order to select only a subset of the data for analysis, the comparer has a `sel()` method which returns a new comparer with the selected data. 

This method allow filtering of the data in several ways:

-   on `observation` by specifying name or index of one or more
    observations
-   on `model` (if more than one is compared) by giving name or index
-   temporal using the `time` (or `start` and `end`) arguments
-   spatial using the `area` argument given as a bounding box or a
    polygon
