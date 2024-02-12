# Vision

[ModelSkill](https://github.com/DHI/modelskill) would like to be your
modelling companion. It should be indispensable good such that you want to use it every time you do a MIKE simulation.


## Objective

We want [ModelSkill](https://github.com/DHI/modelskill) to make it easy
to

-   assess the skill of a model by comparing with *measurements*
-   assess model skill also when result is split on *several files* (2d,
    3d, yearly, \...)
-   compare the skill of different *calibration* runs
-   compare your model with *other models*
-   use a wide range of common evaluation *metrics*
-   create common *plots* such as time series, scatter and taylor
    diagrams
-   do *aggregations* - assess for all observations, geographic areas,
    monthly, \...
-   do *filtering* - assess for a subset of observations, geographic
    areas, \...
-   make *fast* comparisons (optimized code)

And it should be

-   Difficult to make mistakes by verifying input
-   Trustworthy by having \>95% test coverage
-   Easy to install (`$ pip install modelskill`)
-   Easy to get started by providing many notebook examples and
    documentation


## Scope

[ModelSkill](https://github.com/DHI/modelskill) wants to balance general
and specific needs:

- It should be general enough to cover \>90% of MIKE simulations

- It should be general enough to cover generic modelling irrespective
    of software.

- But specific enough to be useful

    - Support dfs files (using [mikeio](https://github.com/DHI/mikeio))
    - Handle circular variables such as wave direction

        
## Limitations

[ModelSkill](https://github.com/DHI/modelskill) does **not** wish to
cover

-   Extreme value analysis
-   Deterministic wave analysis such as crossing analysis
-   Rare alternative file types
-   Rarely used model result types
-   Rare observation types
-   Anything project specific


## Future


### Forecast skill 

It should be possible to compare forecasts with observations using 
forecast lead time as a dimension. Planned 2024. 


### Better support for 3D data

Currently 3D data is supported only as point data and only if data has 
already been extracted from model result files. It should be possible to extract 
date from 3D files directly. Furthermore, vertical columns data should be supported as an observation type with z as a dimension. Planned 2024.


### Web app

Create a web app that wraps this library. 


### Automatic reports

Both static as markdown, docx, pptx and interactive as html. 

