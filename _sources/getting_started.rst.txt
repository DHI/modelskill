.. _getting_started:

Getting started
###############

Workflow
********

The typical fmskill workflow consists of these five steps:

#. Define **ModelResults**
#. Define **Observations**
#. **Associate** observations with ModelResults
#. **Extract** ModelResults at observation positions
#. Do analysis, plotting, etc with a **Comparer**


1. Define ModelResults
======================

The result of a MIKE 21/3 simulation is stored in one or more dfs files. 
The most common formats are .dfsu for distributed data and .dfs0 for 
time series point data. A fmskill *ModelResult* is defined by the 
result file path and a name:

.. code-block:: python

   from fmskill import ModelResult
   mr = ModelResult("HKZN_local_2017.dfsu", name="HKZN_local")

Currently, ModelResult supports .dfs0 and .dfsu files. 
Only the file header is read when the ModelResult object is created. 
The data will be read later. 


2. Define Observations
======================

The next step is to define the measurements to be used for the skill assessment. 
Two types of observation are available: 

* PointObservation
* TrackObservation

Let's assume that we have one point observation and one track observation: 

.. code-block:: python

   from fmskill import PointObservation, TrackObservation
   HKNA = PointObservation("HKNA_Hm0.dfs0", item=0, x=4.2420, y=52.6887, name="HKNA")
   c2 = TrackObservation("Alti_c2_Dutch.dfs0", item=3, name="c2")

In this case both observations are provided as .dfs0 files but pandas 
dataframes are also supported in case data are stored in another file format. 

Both PointObservation and TrackObservation need the path of the data file, 
the item number (or item name) and a name. 

A PointObservation further needs to be initialized with it's x-, y-position. 




3. Associate observations with ModelResults
===========================================

The observations are associated with a model result one by one using the 
``add_observation()`` method like this:


.. code-block:: python

   mr.add_observation(HKNA, item=0)
   mr.add_observation(c2, item=0)   




4. Extract ModelResults at observation positions
================================================

Once the observations have been associated with the model results, 
its very simple to do the extraction which interpolates the model results 
in space and time to the observation points: 

.. code-block:: python

   cc = mr.extract()

The extract method returns a ComparerCollection for further analysis and plotting. 


5. Do analysis, plotting, etc with a Comparer
=============================================

The object returned by the ``extract()`` method is a *comparer*. 
It holds the matched observation and model data and has methods 
for plotting and skill assessment. 

The primary comparer methods are:

* ``skill()`` which returns a pandas dataframe with the skill scores
* ``scatter()`` which shows a scatter density plot of the data


Filtering
---------

Both methods allow filtering of the data in several ways:

* on ``observation`` by specifying name or id of one or more observations
* on ``model`` (if more than one is compared) by giving name or id 
* temporal using the ``start`` and ``end`` arguments
* spatial using the ``area`` argument given as a bounding box or a polygon