# Selecting/filtering data

The primary data filtering method of ModelSkill is the `sel()` method which is accesible on most ModelSkill data structures. The `sel()` method is a wrapper around `xarray.Dataset.sel()` and can be used to select data based on time, location and/or variable. The `sel()` method returns a new data structure of the same type with the selected data.


## TimeSeries data

Point and track timeseries data of both observation and model result kinds are stored in `TimeSeries` objects which uses `xarray.Dataset` as data container. The `sel()` method can be used to select data based on time and returns a new `TimeSeries` object with the selected data.

```python
>>> o = ms.observation('obs.nc', item='waterlevel')
>>> o_1month = o.sel(time=slice('2018-01-01', '2018-02-01'))
```


## Comparer objects

`Comparer` and `ComparerCollection` contain matched data from observations and model results. The `sel()` method can be used to select data based on time, model, quantity or other criteria and returns a new comparer object with the selected data.

```python
>>> cmp = ms.match(o, [m1, m2])
>>> cmp_1month = cmp.sel(time=slice('2018-01-01', '2018-02-01'))
>>> cmp_m1 = cmp.sel(model='m1')
```



## Skill objects

The `skill()` and `mean_skill()` methods return a `SkillTable` object with skill scores from comparing observation and model result data using different metrics (e.g. root mean square error). The data of the `SkillTable` object is stored in a (MultiIndex) `pandas.DataFrame` which can be accessed via the `data` attribute. The `sel()` method can be used to select specific rows and returns a new `SkillTable` object with the selected data.

```python
>>> sk = cmp.skill()
>>> sk_m1 = sk.sel(model='m1')
```


