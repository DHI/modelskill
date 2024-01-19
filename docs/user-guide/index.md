# User Guide

ModelSkill compares model results with observations. The workflow can be split in two phases:

1. [Matching](matching.md) - making sure that observations and model results are in the same space and time
2. Analysis - [plots](plotting.md) and [statistics](skill.md) of the matched data

If the observations and model results are already matched (i.e. are stored in the same data source), 
the `from_matched()` function can be used to go directly to the analysis phase. 
If not, the `match()` function can be used to match the observations and model results in space and time.
