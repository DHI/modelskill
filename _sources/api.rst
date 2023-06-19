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


Connector
---------
.. automodule:: modelskill.connection
	:members: compare

.. autoclass:: modelskill.connection.Connector
	:members:
	:inherited-members:
	:exclude-members: keys, values, get, items, count, index

.. autoclass:: modelskill.connection.PointConnector
	:members:
	:inherited-members:

.. autoclass:: modelskill.connection.TrackConnector
	:members:
	:inherited-members:


Compare
-------
.. autoclass:: modelskill.comparison.PointComparer
	:members:
	:inherited-members:

.. autoclass:: modelskill.comparison.TrackComparer
	:members:
	:inherited-members:

.. autoclass:: modelskill.comparison.ComparerCollection
	:members:
	:inherited-members:
	:exclude-members: keys, values, get, items, count, index


Skill
-------------
.. autoclass:: modelskill.skill.AggregatedSkill
	:members:
	:inherited-members:


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
