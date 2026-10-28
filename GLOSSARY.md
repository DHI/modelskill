# ModelSkill

Vocabulary for evaluating how well a model reproduces observations. File-format and mesh terms follow [mikeio's glossary](https://github.com/DHI/mikeio/blob/main/GLOSSARY.md); where the two overlap, mikeio's wording wins. Observation shapes map to the [CF discrete sampling geometries](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#discrete-sampling-geometries).

## Language

### Core entities

**Observation**:
Measured data from an instrument or station, used as the reference the model is judged against.
_Avoid_: ground truth, target, label

**Model result**:
Simulation output to be judged against observations.
_Avoid_: prediction, forecast, model output

**Comparer**:
One observation paired with one or more model results after matching in space and time.
_Avoid_: comparison, matched dataset

**ComparerCollection**:
A set of comparers, usually one per station, assessed together.
_Avoid_: multi-comparer, comparer set

**Baseline**:
A model result that makes a naive prediction (mean, persistence, climatology), so a real model's skill can be measured against it.
_Avoid_: null model, dummy model

### Geometry types

**Geometry type (gtype)**:
The spatial shape of an observation or model result; it decides which observations and model results can be matched.

**Point**:
A timeseries at one fixed station; the position is given once for the whole series. e.g. a tide gauge. CF `timeSeries`; not CF `point`, which means unconnected values.
_Avoid_: station (a station can hold several points)

**Track**:
Values along a path; each value carries its own position and time, and two values may share a time. e.g. a satellite altimeter pass. CF `trajectory`.
_Avoid_: moving point

**Grid**:
A regular, axis-aligned spatial extent with a value at each cell per timestep. Model results only.

**Flexible mesh** (FM):
An unstructured spatial extent of nodes and elements with a value per element per timestep. Model results only. The geometry concept; the file is **dfsu**.
_Avoid_: unstructured grid

**Vertical**:
Values along z, repeated over time at one fixed station. e.g. CTD casts at a monitoring station. CF `timeSeriesProfile`.
_Avoid_: profile (CF: a single cast), water column, depth profile

**Network**:
Nodes connected by reaches, as in a river or pipe model. Values sit at named locations; nothing is interpolated in space.

**Node**:
A named location in a network, or a break point on a reach.
_Avoid_: junction, vertex

**Reach**:
A stretch of a network between two nodes.
_Avoid_: edge, link, branch

**Break point**:
A position along a reach, given as (reach, chainage), where the model computes values.
_Avoid_: breakpoint, chainage point

### Data shapes

**Timeseries**:
Values in time at one fixed station (a point). One variable at a time.

**Field**:
Values over a spatial extent (grid or flexible mesh) at each timestep. Matching extracts a timeseries from a field at each observation location.

**Quantity**:
A physical quantity with its unit, e.g. *Water Level [m]*. Only observations and model results of the same quantity can be matched.

### Skill assessment

**Skill**:
How well a model reproduces the observations, broken down per observation and model.
_Avoid_: performance

**Score**:
One number summarising skill for one metric, weighted across time, observations and variables.

**Metric**:
A formula evaluated on matched observation–model pairs, e.g. bias, RMSE, NSE.

**SkillTable**:
Skill metrics grouped by observation, model and variable.

**SkillGrid**:
Skill metrics binned in horizontal space.

**SkillProfile**:
Skill metrics binned in z.
