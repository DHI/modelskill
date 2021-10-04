.. _api:

API Documentation
=================


Model
-----
.. autoclass:: fmskill.model.ModelResult
	:members:

.. autoclass:: fmskill.model.dfs.DfsModelResult
	:members:

.. autoclass:: fmskill.model.dfs.DfsModelResultItem
	:members:

Observation
-----------
.. autoclass:: fmskill.observation.PointObservation
	:members:
	:inherited-members:

.. autoclass:: fmskill.observation.TrackObservation
	:members:
	:inherited-members:


Data
----
.. autoclass:: fmskill.data.DMIOceanObsRepository
	:members:

.. autoclass:: fmskill.data.DHIAltimetryRepository
	:members:

.. autoclass:: fmskill.data.altimetry.AltimetryData
	:members:


Connector
---------
.. automodule:: fmskill.connection
	:members: compare

.. autoclass:: fmskill.connection.Connector
	:members:
	:inherited-members:
	:exclude-members: keys, values, get, items, count, index

.. autoclass:: fmskill.connection.PointConnector
	:members:
	:inherited-members:

.. autoclass:: fmskill.connection.TrackConnector
	:members:
	:inherited-members:


Compare
-------
.. autoclass:: fmskill.comparison.PointComparer
	:members:
	:inherited-members:

.. autoclass:: fmskill.comparison.TrackComparer
	:members:
	:inherited-members:

.. autoclass:: fmskill.comparison.ComparerCollection
	:members:
	:inherited-members:
	:exclude-members: keys, values, get, items, count, index


Skill
-------------
.. autoclass:: fmskill.skill.AggregatedSkill
	:members:
	:inherited-members:


Spatial Skill
-------------
.. autoclass:: fmskill.spatial.SpatialSkill
	:members:


Metrics
-------
.. autosummary:: 
	:nosignatures:
	
	fmskill.metrics.bias	
	fmskill.metrics.root_mean_squared_error
	fmskill.metrics.rmse
	fmskill.metrics.urmse
	fmskill.metrics.mean_absolute_error
	fmskill.metrics.mae
	fmskill.metrics.mean_absolute_percentage_error
	fmskill.metrics.mape
	fmskill.metrics.nash_sutcliffe_efficiency
	fmskill.metrics.nse
	fmskill.metrics.model_efficiency_factor
	fmskill.metrics.mef
	fmskill.metrics.scatter_index
	fmskill.metrics.si
	fmskill.metrics.corrcoef
	fmskill.metrics.cc
	fmskill.metrics.spearmanr
	fmskill.metrics.rho
	fmskill.metrics.r2
	fmskill.metrics.lin_slope
	fmskill.metrics.wilmott
	fmskill.metrics.hit_ratio
	
.. automodule:: fmskill.metrics
	:members:
