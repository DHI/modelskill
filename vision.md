# Vision

[ModelSkill](https://github.com/DHI/modelskill) wishes to be your modelling assistant. It should be useful enough for you to use every time you do a MIKE 21/3 simulation. 


## Objective

We want [ModelSkill](https://github.com/DHI/modelskill) to make it easy to 

* assess the skill of a model by comparing with *measurements*
* assess model skill also when result is split on *several files* (2d, 3d, yearly, ...)
* compare the skill of different *calibration* runs
* compare your model with *other models*
* use a wide range of common evaluation *metrics* 
* create common *plots* such as time series, scatter and taylor diagrams
* do *aggregations* - assess for all observations, geographic areas, monthly, ...
* make *fast* comparisons (optimized code)

And it should be 

* Difficult to make mistakes by verifying input 
* Trustworthy by having >95% test coverage 
* Easy to install (`pip install modelskill`)
* Easy to get started by providing many notebook examples and documentation


## Scope 

[ModelSkill](https://github.com/DHI/modelskill) wants to balance general and specific needs: 

* It should be general enough to cover >90% of MIKE FM simulations
* But specific enough to be *useful*
    - Primarily support dfs files (using [mikeio](https://github.com/DHI/mikeio))
    - Handle circular variables such as wave direction
    - Handle vector variables such as u- and v-components of current
    - Tidal analysis



## Limitations

[ModelSkill](https://github.com/DHI/modelskill) does **not** wish to cover 

* Extreme value analysis
* Deterministic wave analysis such as crossing analysis
* Rare alternative file formats
* Rarely used model result types 
* Rare observation types
* Anything project specific

## Future

### Web app
Create a web app that wraps this library 

