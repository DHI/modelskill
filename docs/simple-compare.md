Simple time series comparison
=============================

If all you need to do is to compare two point time series, the workflow
is very simple and described below. The general many-to-many comparison
is decribed in the [getting started guide](getting-started).

Workflow
--------

The simplified modelskill workflow consists of these four steps:

1.  Specify **model result**
2.  Specify **observation**
3.  **compare()**
4.  Analysis and plotting

### 1. Specify model result

The model result can be either a dfs0 or a DataFrame.

```python
import mikeio
fn_mod = '../tests/testdata/SW/ts_storm_4.dfs0'
```

### 2. Specify Observation

The observation can be either a dfs0, a DataFrame or a PointObservation
object.

```python
fn_obs = '../tests/testdata/SW/eur_Hm0.dfs0'
```

### 3. compare()

The [compare()](modelskill.connection.compare) method will
interpolate the modelresult to the time of the observation and return an
object that can be used for analysis and plotting

```python
import modelskill
c = modelskill.compare(fn_obs, fn_mod, mod_item=0)
```

### 4. Analysis and plotting

The returned
[PointComparer](modelskill.comparison.PointComparer) can make
scatter plots, skill assessment, time series plots etc.

```python
>>> c.plot.timeseries()
```

![image](images/ts_plot.png)

```python
>>> c.plot.scatter()
```

![image](images/scatter_plot.png)

```python
>>> c.skill()
            n     bias      rmse     urmse       mae        cc        si        r2
observation
eur_Hm0      66  0.05321  0.229957  0.223717  0.177321  0.967972  0.081732  0.929005
```