.. _api:

API Documentation
=================


Model
-----
.. autoclass:: fmskill.model.ModelResult
	:members:

.. autoclass:: fmskill.model.ModelResultCollection
	:members:
	:inherited-members:

Observation
-----------
.. autoclass:: fmskill.observation.PointObservation
	:members:
	:inherited-members:

.. autoclass:: fmskill.observation.TrackObservation
	:members:
	:inherited-members:

Compare
-------
.. autoclass:: fmskill.compare.PointComparer
	:members:
	:inherited-members:

.. autoclass:: fmskill.compare.TrackComparer
	:members:
	:inherited-members:

.. autoclass:: fmskill.compare.ComparerCollection
	:members:
	:inherited-members:
	:exclude-members: keys, values, get, items

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
	fmskill.metrics.spearmanr
	fmskill.metrics.rho
	fmskill.metrics.r2
	fmskill.metrics.lin_slope
	
.. automodule:: fmskill.metrics
	:members:
