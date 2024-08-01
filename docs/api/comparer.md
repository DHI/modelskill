# Comparer

The `Comparer` class is the main class of the ModelSkill package. It is returned by [`match()`](matching.md/#modelskill.match), [`from_matched()`](matching.md/#modelskill.from_matched) or as an element in a [`ComparerCollection`](comparercollection.md). It holds the *matched* observation and model data for a *single* observation and has methods for plotting and skill assessment.

Main functionality:

* selecting/filtering data
    - [`sel()`](#modelskill.Comparer.sel)
    - [`query()`](#modelskill.Comparer.query)
* skill assessment
    - [`skill()`](#modelskill.Comparer.skill)
    - [`gridded_skill()`](#modelskill.Comparer.gridded_skill) (for track observations)
* plotting
    - [`plot.timeseries()`](#modelskill.comparison._comparer_plotter.ComparerPlotter.timeseries)
    - [`plot.scatter()`](#modelskill.comparison._comparer_plotter.ComparerPlotter.scatter)
    - [`plot.kde()`](#modelskill.comparison._comparer_plotter.ComparerPlotter.kde)
    - [`plot.qq()`](#modelskill.comparison._comparer_plotter.ComparerPlotter.qq)
    - [`plot.hist()`](#modelskill.comparison._comparer_plotter.ComparerPlotter.hist)
    - [`plot.box()`](#modelskill.comparison._comparer_plotter.ComparerPlotter.box)
* load/save/export data
    - [`load()`](#modelskill.Comparer.load)
    - [`save()`](#modelskill.Comparer.save)
    - [`to_dataframe()`](#modelskill.Comparer.to_dataframe)



::: modelskill.Comparer


::: modelskill.comparison._comparer_plotter.ComparerPlotter