<img src="https://raw.githubusercontent.com/DHI/modelskill/main/images/logo/modelskill.svg" width="300">

# ModelSkill: compare model results with observations in Python
 ![Python version](https://img.shields.io/pypi/pyversions/modelskill.svg)
 [![Full test](https://github.com/DHI/modelskill/actions/workflows/full_test.yml/badge.svg)](https://github.com/DHI/modelskill/actions/workflows/full_test.yml)
[![PyPI version](https://badge.fury.io/py/modelskill.svg)](https://badge.fury.io/py/modelskill)
![OS](https://img.shields.io/badge/OS-Windows%20%7C%20Linux-blue)
![Downloads](https://img.shields.io/pypi/dm/modelskill)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/DHI/modelskill/blob/main/LICENSE)

Match observations with model results, calculate skill metrics, and make plots and reports.

ModelSkill is primarily used for [MIKE](https://www.dhigroup.com/technologies/mikepoweredbydhi) models, but other models can be evaluated as well. It is meant to be your companion during model setup, calibration, and validation and reporting.

## Requirements

* Windows or Linux operating system
* Python x64 3.12 - 3.14

## Installation

```bash
pip install modelskill
```

Or the development version:

```bash
pip install https://github.com/DHI/modelskill/archive/main.zip
```

## Getting started

Define model results and observations:

```python
>>> import modelskill as ms
>>> mr = ms.DfsuModelResult("HKZN_local_2017_DutchCoast.dfsu", name="HKZN_local", item=0)
>>> HKNA = ms.PointObservation("HKNA_Hm0.dfs0", item=0, x=4.2420, y=52.6887, name="HKNA")
>>> EPL = ms.PointObservation("eur_Hm0.dfs0", item=0, x=3.2760, y=51.9990, name="EPL")
>>> c2 = ms.TrackObservation("Alti_c2_Dutch.dfs0", item=3, name="c2")
```

Match them in space and time, extracting model data at the observation positions:

```python
>>> cc = ms.match([HKNA, EPL, c2], mr)
```

The resulting `ComparerCollection`, cc, is the starting point for skill assessment and plotting:

```python
>>> cc.skill().round(2)
               n  bias  rmse  urmse   mae    cc    si    r2
observation
HKNA         386 -0.20  0.36   0.29  0.26  0.97  0.09  0.90
EPL           67 -0.07  0.22   0.21  0.19  0.97  0.08  0.93
c2           113 -0.00  0.35   0.35  0.29  0.98  0.13  0.90
>>> cc.plot.scatter()
>>> cc["HKNA"].plot.timeseries(backend="plotly")
```

See the [user guide](https://dhi.github.io/modelskill/user-guide/getting-started.html) for more.

## Where can I get help?
* Documentation - [https://dhi.github.io/modelskill/](https://dhi.github.io/modelskill/)
* Examples - [https://dhi.github.io/modelskill/examples/](https://dhi.github.io/modelskill/examples/)
* General help, new ideas and feature requests - [GitHub Discussions](https://github.com/DHI/modelskill/discussions)
* Bugs - [GitHub Issues](https://github.com/DHI/modelskill/issues)

## Testing

ModelSkill is tested extensively, with an overall statement coverage of ~90%. The test suite runs on every pull request against Python 3.12 and 3.14, and on a schedule on both Linux and Windows.

```bash
uv run pytest --cov=modelskill
```

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](https://github.com/DHI/modelskill/blob/main/CONTRIBUTING.md). Key architectural decisions are documented as [ADRs](https://github.com/DHI/modelskill/tree/main/adr).

## License

[MIT](https://github.com/DHI/modelskill/blob/main/LICENSE)
