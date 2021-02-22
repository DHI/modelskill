# Vision

[mikefm-skill](https://github.com/DHI/mikefm-skill) wishes to be your modelling assistant. It should useful enough for you to use every time you do a MIKE 21/3 simulation. 


## Objective

We want [mikefm-skill](https://github.com/DHI/mikefm-skill) to make it

* Easy to assess the skill of a model by comparing with measurements
* Easy to assess model skill also when result is split on several files (2d, 3d, yearly, ...)
* Easy to compare the skill of different calibration runs
* Easy to compare your model with other models (and climatology)
* Easy to use a wide range of common evaluation metrics 
* Easy to create common plots such as time series, scatter and taylor diagrams
* Easy to do aggregations - assess for all observations, geographic areas, monthly, ...
* Fast to make comparisons (optimized code)
* Difficult to make mistakes by verifying input 
* Trustworthy by having >95% test coverage 
* Easy to install (from pypi and conda)
* Easy to get started by providing many notebook examples and documentation


## Scope 

[mikefm-skill](https://github.com/DHI/mikefm-skill) wants to balance general and specific needs: 

* It should be general enough to cover >90% of MIKE FM simulations
* But specific enough to be *useful*
    - Primarily support dfs files (using [mikeio](https://github.com/DHI/mikeio))
    - Handle circular variables such as wave direction
    - Handle vector variables such as u- and v-components of current
    - Tidal analysis



## Limitations

[mikefm-skill](https://github.com/DHI/mikefm-skill) does **not** wish to cover 

* Extreme value analysis
* Forecast skill assessments
* Deterministic wave analysis such as crossing analysis
* Alternative file types 
* Rarely used model result types 
* Rare observation types
* Anything project specific



## Future

### Automatic reports
Both static as markdown, docx, pptx and interactive as html


### Web app
Create a web app that wraps this library 


### Interface to observation APIs
Easy to get observation data from [DHI's altimetry portal](https://altimetry.dhigroup.com/), CMEMS, etc. 


### Interface to alternative models
Should be easy to compare your model to publically available alternative e.g. from CMEMS or NOAA. Or from DHI's DataLink. 