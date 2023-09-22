.. _api:

API Documentation
=================


Model
-----
.. autoclass:: modelskill.model.PointModelResult
	:members:

.. autoclass:: modelskill.model.TrackModelResult
	:members:

.. autoclass:: modelskill.model.DfsuModelResult
	:members:

.. autoclass:: modelskill.model.GridModelResult
	:members:


Observation
-----------
.. autoclass:: modelskill.observation.PointObservation
	:members:
	:inherited-members:

.. autoclass:: modelskill.observation.TrackObservation
	:members:
	:inherited-members:


Quantity
--------
.. autoclass:: modelskill.types.Quantity
	:members:


Compare
-------

.. autofunction:: modelskill.compare

.. autofunction:: modelskill.from_matched

.. autoclass:: modelskill.comparison.Comparer
	:members:

.. autoclass:: modelskill.comparison._comparer_plotter.ComparerPlotter
	:members:

.. autoclass:: modelskill.comparison.ComparerCollection
	:members:
	:inherited-members:
	:exclude-members: keys, values, get, items, count, index

.. autoclass:: modelskill.comparison._collection_plotter.ComparerCollectionPlotter
	:members:


Skill
-------------
.. autoclass:: modelskill.skill.AggregatedSkill
	:members:
	:inherited-members:

.. autoclass:: modelskill.skill.AggregatedSkillPlotter
	:members:
	

Spatial Skill
-------------
.. autoclass:: modelskill.spatial.SpatialSkill
	:members:


Metrics
-------
.. autosummary:: 
	:nosignatures:
	
	modelskill.metrics.bias	
	modelskill.metrics.max_error
	modelskill.metrics.root_mean_squared_error
	modelskill.metrics.rmse
	modelskill.metrics.urmse
	modelskill.metrics.mean_absolute_error
	modelskill.metrics.mae
	modelskill.metrics.mean_absolute_percentage_error
	modelskill.metrics.mape
	modelskill.metrics.nash_sutcliffe_efficiency
	modelskill.metrics.nse
	modelskill.metrics.kling_gupta_efficiency
	modelskill.metrics.kge
	modelskill.metrics.model_efficiency_factor
	modelskill.metrics.mef
	modelskill.metrics.scatter_index
	modelskill.metrics.si
	modelskill.metrics.corrcoef
	modelskill.metrics.cc
	modelskill.metrics.spearmanr
	modelskill.metrics.rho
	modelskill.metrics.r2
	modelskill.metrics.lin_slope
	modelskill.metrics.willmott
	modelskill.metrics.hit_ratio
	
.. automodule:: modelskill.metrics
	:members:

Plot
----

.. autofunction:: modelskill.plotting.temporal_coverage

.. autofunction:: modelskill.plotting.spatial_overview

.. autofunction:: modelskill.plotting.wind_rose


Settings
--------

.. autofunction:: modelskill.get_option

.. autofunction:: modelskill.set_option

.. autofunction:: modelskill.reset_option

.. autofunction:: modelskill.load_style
	