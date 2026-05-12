# ADR-011: VerticalModelResult ingests pre-extracted columns

**Status**: Accepted

**Date**: 2026-04-28

## Context

ModelSkill's other model result types follow a "give us the source file, we'll extract at the observation location" pattern: `GridModelResult` and `DfsuModelResult` accept the full simulation output and extract a `PointModelResult` at the observation's (x, y) on demand. This auto-extract pattern keeps the user's workflow short and notebook-friendly.

For a `VerticalModelResult` the equivalent operation is "extract one column from a 3D dfsu at a single (x, y)" — picking the mesh element nearest the observation, then reconstructing all sigma-z layers along the time axis. In practice this operation is **slow** (mesh navigation, layer reconstruction, time loop) and does not fit an interactive notebook session where the user iterates on plots and skill numbers.

## Decision

`VerticalModelResult` ingests **pre-extracted single-column data only** — dfs0, `pandas.DataFrame`, or `xarray.Dataset`. Extraction from a 3D dfsu (or other 3D source) is the user's responsibility, performed offline with mikeio (or equivalent) and written to a dfs0.

The matching pipeline (`match()`) consequently does **not** check spatial agreement between a `VerticalObservation` and a `VerticalModelResult`. The model column's (x, y) is the chosen mesh element center, not the observation's instrument position; a small disagreement is expected, not a bug.

## Alternatives Considered

**Auto-extract on construction (parallel to Grid/Dfsu).** Would block notebook sessions on every `VerticalModelResult(...)` call. Iterating on a 3D dfsu workflow would be unusable.

**Lazy extraction with caching.** Adds machinery (cache keying, invalidation, on-disk vs in-memory) for the first-call wait that still has to happen. Cost not justified by current usage.

**Add a `Dfsu3DModelResult` type that produces `VerticalModelResult` via `extract_column(x, y)`.** Orthogonal to this decision — does not change what `VerticalModelResult` ingests; only where the column originates. Can be layered on later without revisiting this ADR.

## Consequences

- **Friction for new users.** Anyone with a 3D dfsu must run a separate mikeio script before touching modelskill. Documented as the standard workflow; not a blocker for the target users (MIKE practitioners who already use mikeio).
- **`VerticalModelResult` stays simple.** One ingestion path (single column in), one storage shape, one matching routine.
- **No spatial check at match time.** Obs (x, y) and model (x, y) legitimately differ by up to half a mesh element. A strict equality check would over-trigger under normal usage; users wanting a check can compare manually.
- **3D extraction can be added later as a separate type** (`Dfsu3DModelResult` or similar) without changing `VerticalModelResult`'s contract. Auto-extraction remains a non-goal *for `VerticalModelResult`*; if added, it lives on the 3D source type.
