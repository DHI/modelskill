# API Documentation

Obtain a comparer object in one of the following ways: 

- From matched data with [from_matched()](matching.md#modelskill.from_matched)
- After defining [observation](observation/index.md)s and [model result](model/index.md)s using the [match()](matching.md#modelskill.match) function.
- From a config file with [from_config()](matching.md#modelskill.from_config)

Do analysis and plotting with the returned [Comparer](comparer.md#modelskill.Comparer) (a single observation) or [ComparerCollection](comparercollection.md#modelskill.comparison.ComparerCollection) (multiple observations):

- [skill()](comparercollection.md#modelskill.comparison.ComparerCollection.skill) - returns a [SkillTable](skill.md) with the skill scores
- plot using the various plot methods of the comparer objects
    * `plot.scatter()`
    * `plot.timeseries()`
    * `plot.kde()`
    * `plot.qq()`
    * `plot.hist()`