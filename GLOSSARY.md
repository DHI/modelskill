# ModelSkill

Vocabulary for evaluating how well a model reproduces observations. File-format and mesh terms follow [mikeio's glossary](https://github.com/DHI/mikeio/blob/main/CONTEXT.md); where the two overlap, mikeio's wording wins.

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
A fixed (x, y) location with one z, holding a timeseries. e.g. a tide gauge.

**Track**:
A moving (x, y) location with one z, holding a timeseries. e.g. a satellite altimeter.

**Grid**:
A regular, axis-aligned spatial extent with a value at each cell per timestep. Model results only.

**Flexible mesh** (FM):
An unstructured spatial extent of nodes and elements with a value per element per timestep. Model results only. The geometry concept; the file is **dfsu**.
_Avoid_: unstructured grid

**Vertical**:
A fixed (x, y) location with values varying in z and time: a water column at one station. e.g. a CTD cast.
_Avoid_: profile, water column, depth profile

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
Values in time at one location, fixed (point) or moving (track). One variable at a time.

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
