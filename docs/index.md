![image](https://raw.githubusercontent.com/DHI/modelskill/main/images/logo/modelskill.svg)


# ModelSkill: Assess the skill of your MIKE model
![Python version](https://img.shields.io/pypi/pyversions/modelskill.svg)
![PyPI version](https://badge.fury.io/py/modelskill.svg)
    
Compare results from MIKE simulations with observations.
[ModelSkill](https://github.com/DHI/modelskill) would like to be your
companion during the different phases of a MIKE modelling workflow.


## Installation

ModelSkill is available as open-source on PyPI and can be installed with pip:

```bash
$ pip install modelskill
```

ModelSkill is compatible with Python 3.8 and later versions on Windows and Linux.


## Getting started

Are your observations and model results already matched? 

```python
import modelskill as ms
cmp = ms.from_matched("matched_data.dfs0", obs_item="obs_WL", mod_item="WL")
cmp.skill()
```

Or do you need to match the observations and results first?

```python
import modelskill as ms
o = ms.PointObservation("obs.dfs0", item="obs_WL")
mr = ms.PointModelResult("model.dfs0", item="WL")
cmp = ms.match(o, mr)
cmp.skill()
```

Read more in the [Getting started guide](getting-started.md) or in the [overview](overview.md) of the package.


## Resources

- [Documentation](https://dhi.github.io/modelskill/) (this site)
- [Getting started guide](getting-started.md)
- [Example notebooks](https://nbviewer.jupyter.org/github/DHI/modelskill/tree/main/notebooks/)
- [PyPI](https://pypi.org/project/modelskill/)
- [Source code](https://github.com/DHI/modelskill/)

