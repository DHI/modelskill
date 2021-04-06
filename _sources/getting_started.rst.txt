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


Define ModelResults
===================


Define Observations
===================


Associate observations with ModelResults
========================================
The observations are associated with a model result one by one using the 
``add_observation()`` method like this:


.. code-block:: python

   mr.add_observation(x,y,z)




Extract ModelResults at observation positions
=============================================

Once the observations have been associated with the model results, 
its very simple to do the extraction which interpolates the model results 
in space and time to the observation points: 

.. code-block:: python

   cc = mr.extract()



Do analysis, plotting, etc with a Comparer
==========================================

The object returned by the ``extract()`` method is a Comparer. 
It holds the matched observation and model data and has methods for plotting and skill assessment. 
The primary methods are:

* ``skill()`` which returns a pandas dataframe with the skill scores
* ``scatter()`` which shows a scatter density plot of the data

Both methods allow filtering of the data in several ways:

* on ``observation`` by specifying name or id of one or more observations
* on ``model`` (if more than one is compared) by giving name or id 
* temporal using the ``start`` and ``end`` arguments
* spatial using the ``area`` argument given as a bounding box or a polygon