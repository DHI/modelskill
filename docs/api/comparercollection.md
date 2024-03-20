# ComparerCollection

The `ComparerCollection` is one of the main objects of the `modelskill` package. It is a collection of [`Comparer`](comparer.md) objects and created either by the [`match()`](matching.md/#modelskill.match) method, by passing a list of Comparers to the [`ComparerCollection`](comparercollection.md/#modelskill.ComparerCollection) constructor, or by reading a config file using the [`from_config()`](matching.md/#modelskill.from_config) function.

Main functionality:

* selecting/filtering data
    - `__get_item__()` - get a single Comparer, e.g., `cc[0]` or `cc['obs1']` 
    - [`sel()`](#modelskill.ComparerCollection.sel)
    - [`query()`](#modelskill.ComparerCollection.query)
* skill assessment
    - [`skill()`](#modelskill.ComparerCollection.skill)
    - [`mean_skill()`](#modelskill.ComparerCollection.mean_skill)
    - [`gridded_skill()`](#modelskill.ComparerCollection.gridded_skill) (for track observations)
* plotting
    - [`plot.scatter()`](#modelskill.comparison._collection_plotter.ComparerCollectionPlotter.scatter)
    - [`plot.kde()`](#modelskill.comparison._collection_plotter.ComparerCollectionPlotter.kde)
    - [`plot.hist()`](#modelskill.comparison._collection_plotter.ComparerCollectionPlotter.hist)
* load/save/export data
    - [`load()`](#modelskill.ComparerCollection.load)
    - [`save()`](#modelskill.ComparerCollection.save)



::: modelskill.ComparerCollection

::: modelskill.comparison._collection_plotter.ComparerCollectionPlotter
