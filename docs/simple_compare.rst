.. _simple_compare:

Simple time series comparison
#############################

If all you need to do is to compare two point time series, the workflow is 
very simple and described below. The general many-to-many comparison is decribed 
in the `getting started guide <getting_started.html>`_.


Workflow
********

The simplified modelskill workflow consists of these four steps:

#. Specify **model result**
#. Specify **observation**
#. **compare()**
#. Analysis and plotting


1. Specify model result
=======================

The model result can be either a dfs0 or a DataFrame.

.. code-block:: python

    import mikeio
    fn_mod = '../tests/testdata/SW/ts_storm_4.dfs0'


2. Specify Observation
======================
The observation can be either a dfs0, a DataFrame or a PointObservation object. 

.. code-block:: python

    fn_obs = '../tests/testdata/SW/eur_Hm0.dfs0'


3. compare()
============
The `compare() <api.html#modelskill.connection.compare>`_ method will interpolate the modelresult to the time of the observation
and return an object that can be used for analysis and plotting

.. code-block:: python

    import modelskill
    c = modelskill.compare(fn_obs, fn_mod, mod_item=0)


4. Analysis and plotting
========================

The returned `PointComparer <api.html#modelskill.comparison.PointComparer>`_ can make
scatter plots, skill assessment, time series plots etc.


.. code-block:: python

    >>> c.plot.timeseries()

.. image:: images/ts_plot.png


.. code-block:: python

    >>> c.plot.scatter()

.. image:: images/scatter_plot.png

.. code-block:: python

    >>> c.skill()
                n     bias      rmse     urmse       mae        cc        si        r2
    observation
    eur_Hm0      66  0.05321  0.229957  0.223717  0.177321  0.967972  0.081732  0.929005


