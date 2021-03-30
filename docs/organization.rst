.. _organization:

Getting started
###############

Workflow




Core concepts
*************

fmskill is an object-oriented package built around a few basic concepts:

* ModelResult: defined by a MIKE FM output (.dfsu or .dfs0 file), observations can be added to a ModelResult 
* Observation: e.g. point or track observation
* Metric: can measure the "distance" between a model result and an observation (e.g. bias and rmse)
* Comparer: contains observations and model data interpolated to observation positions and times, can plot and show statistics

