# Comparer

The `Comparer` class is the main class of the ModelSkill package. It holds the *matched* observation and model data for a *single* observation and has methods for plotting and skill assessment.

Main functionality:

* selecting/filtering data
    - `sel()`
    - `query()`
* skill assessment
    - `skill()`
    - `gridded_skill()` (for track observations)
* plotting
    - `plot.timeseries()`
    - `plot.scatter()`
    - `plot.kde()`
    - `plot.qq()`
    - `plot.hist()`
    - `plot.box()`
* load/save/export data
    - `load()`
    - `save()`
    - `to_dataframe()`



::: modelskill.Comparer


::: modelskill.comparison._comparer_plotter.ComparerPlotter