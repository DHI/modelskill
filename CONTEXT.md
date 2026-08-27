# ModelSkill

Internal domain glossary for ModelSkill — a Python package for evaluating model skill by comparing simulation output with observations, primarily for MIKE 21/3 (Flexible Mesh) hydrodynamic and oceanographic models. The user-facing version lives in `docs/user-guide/terminology.qmd`; this file is the dev-facing canonical reference and additionally documents internal/alpha vocabulary that is not yet stable enough for the public docs.

Upstream vocabulary (file formats, geometry concepts, layered meshes) lives in [mikeio's CONTEXT.md](../mikeio/CONTEXT.md). Where the two glossaries overlap, mikeio is authoritative — modelskill aligns rather than diverges.

## Language

### Core entities

**Observation**:
Measured data from a real instrument or station, used as the reference truth. Concrete types: `PointObservation`, `TrackObservation` (plus `VerticalObservation`, alpha).
_Avoid_: ground truth, target, label.

**ModelResult**:
Simulation output to be evaluated against an Observation. Concrete types: `PointModelResult`, `TrackModelResult`, `GridModelResult`, `DfsuModelResult`, `DummyModelResult` (plus `VerticalModelResult`, alpha).
_Avoid_: prediction, forecast, model output.

**Comparer**:
The pairing of one Observation with one or more ModelResults after spatial and temporal matching. Holds the matched data alongside the raw model data.
_Avoid_: comparison, matched dataset.

**ComparerCollection**:
A collection of Comparers, typically one per station, used for cross-station skill assessment.
_Avoid_: multi-comparer, comparer set.

**Baseline**:
A reference ModelResult representing a naive prediction (mean, persistence, climatology). Realized as `DummyModelResult`. Anchors skill scores like NSE that quantify "is the real model better than predicting nothing clever".
_Avoid_: null model, dummy model (use "Baseline" in prose; `DummyModelResult` is the class).

### Geometry types (gtype)

**Geometry type (gtype)**:
The label describing the spatial structure of an Observation or ModelResult. Determines match-compatibility. Values: `point`, `track`, `grid`, `unstructured` (flexible mesh), `vertical` (alpha), `node` / `reach` (future). Accessible via `.gtype`.

**Point**:
A fixed (x, y) location, single z, timeseries of values. e.g. tide gauge, single-depth mooring.

**Track**:
A moving (x, y) location, single z, timeseries of values. e.g. satellite altimeter, ship transect.

**Grid**:
A regular axis-aligned spatial extent with a value at each (x, y) per timestep. ModelResult-only (no GridObservation). Backed by `GridModelResult` (xarray / nc / dfs2).

**Flexible Mesh** (FM):
An unstructured spatial extent of Nodes + Elements with a value per Element per timestep. ModelResult-only. Backed by `DfsuModelResult` (dfsu files). Use **flexible mesh** for the geometry concept and **dfsu** for the file format — never "unstructured grid". Aligns with mikeio's vocabulary.

**Vertical** (alpha):
A fixed (x, y) location with values varying along z and time — a water column at one station. e.g. CTD cast, moored profiler, modeled column extracted from a 3D dfsu.
_Avoid_: profile, water column, column, depth profile (all overloaded).

### Skill assessment

**Skill**:
The ability of a model to reproduce observations. As an API, `Comparer.skill()` returns a `SkillTable` of metrics grouped by (observation, model, variable). Conceptually a Comparer-level concept; also available on ComparerCollection.

**Score**:
A single numerical value summarizing model performance for one metric, computed as a weighted average across all time-steps, observations and variables. Conceptually a ComparerCollection-level concept; also available on `Comparer.score` as the degenerate single-observation case.

**Metric**:
A mathematical expression evaluated on matched (obs, model) pairs (bias, RMSE, R², NSE, …). Defined in `modelskill.metrics`. Both circular and scalar metrics are supported.

**SkillTable**:
Skill metrics aggregated by categorical groupings (model, observation, variable). DataFrame-like.

**SkillGrid**:
Skill metrics binned in horizontal space (x, y).

**SkillProfile** (alpha):
Skill metrics binned in z. Single-station today; will gain an `observation` dim for multi-station before release.

### Physical quantities

**Quantity**:
A physical quantity (name + unit, e.g. *Water Level [m]*) attached to an Observation or ModelResult. Compatibility gates matching — water level cannot be matched against wind speed. When reading dfs files via mikeio, derived from `EUMType` + `EUMUnit`. Class: `Quantity`.

**Field**:
Data defined over a spatial extent (Grid or Flexible Mesh) with a value at each location and timestep — i.e. the gtype=grid or gtype=unstructured shape. Contrasts with **timeseries** (point/track). Not directly comparable to observations; `match()` extracts a timeseries per observation location before producing a Comparer.

**Timeseries**:
A sequence of values in time at a single location (point) or a single moving location (track). Univariate by convention in ModelSkill; multivariate timeseries are assessed one variable at a time.

## Relationships

- An **Observation** and a **ModelResult** of compatible **gtype** match into a **Comparer** (or **ComparerCollection** for multiple observations).
- A **Field** (Grid / FM ModelResult) is *not* directly comparable to an Observation; `match()` extracts a **Timeseries** at each observation location.
- **Skill** is naturally per-Comparer (per-(obs, model) breakdown); **Score** is naturally per-ComparerCollection (cross-observation weighted average). Both methods exist on both classes for convenience.
- A **Comparer** of `gtype="vertical"` exposes a `.vertical` accessor for column-specific operations (`skill`, `plot.profile`, `plot.hovmoller`). *Alpha.*
- A **ComparerCollection** of vertical members will expose `cc.vertical.skill()` (release blocker, follow-up PR) returning a **SkillProfile** with `(z, observation, model)` dims. *Alpha.*

## Conventions

**z-axis direction**:
Default `positive="up"` (MIKE 3 convention — z=0 at datum, below-surface negative). Read from `z.attrs["positive"]` if set; user can override via the `positive` constructor kwarg. Plotter inverts the y-axis iff `positive="down"`. mikeio-sourced datasets do not currently carry this attribute.

**Flexible Mesh vs dfsu**:
*Flexible mesh* (or FM) is the geometry/engine concept; *dfsu* is the binary file format (extension `.dfsu`). Class names anchored to the file use `Dfsu*`; prose about the geometry uses "flexible mesh". Aligns with mikeio.

## Flagged ambiguities

- **"performance" vs "skill"**: Treated as aliases in user docs (terminology.qmd folds "performance" into Skill). Internally, prefer "skill".
- **"unstructured" vs "flexible mesh"**: The code's `GeometryType.UNSTRUCTURED` enum value is a legacy name. Prose and docs say "flexible mesh"; the enum rename was deferred (flag in API reviews, do not change unprompted).
- **(x, y) on a VerticalObservation vs VerticalModelResult**: legitimately differ. Obs holds the true CTD/instrument position; model holds the nearest mesh element center. No equality check is performed at match time — the mismatch is expected, not a bug. *Alpha.*
- **"profile"**: Rejected as an entity name — overloaded across `SkillProfile`, `plot.profile()`, Taylor profiles, and (informally) CTD output. The entity is **Vertical**; "profile" survives only when describing the *output* (e.g., `SkillProfile` reads as "skill as a function of depth"). *Alpha.*
- **3D dfsu → VerticalModelResult extraction**: not provided in-package. Extraction is an offline preprocessing step because it is slow (mesh navigation, sigma-z layer reconstruction). Users extract one column with mikeio, write a dfs0, and pass that to `VerticalModelResult`. *Alpha.*

## Example dialogue

> **Dev:** "I have a tide-gauge timeseries and a dfsu of surface elevation. What's the workflow?"
>
> **Maintainer:** "Wrap the gauge as a `PointObservation` and the dfsu as a `DfsuModelResult` (gtype `unstructured`, i.e. flexible mesh). Call `match()` — it'll extract a timeseries from the mesh at the gauge location and hand you back a `Comparer`. Then `cmp.skill()` gives you a `SkillTable` of metrics, or `cmp.score(metric=...)` for a single number."
>
> **Dev:** "And if I have ten gauges?"
>
> **Maintainer:** "`match()` with a list of observations returns a `ComparerCollection`. `cc.score(...)` is the natural one-number summary — it weights across observations. `cc.skill()` still gives you the per-observation breakdown."
>
> **Dev:** "Is my model better than just predicting the mean?"
>
> **Maintainer:** "Add a `DummyModelResult` as a baseline alongside your real model, run `match()`, and compare scores. NSE is the metric that's literally defined as 'how much better than the mean'."
