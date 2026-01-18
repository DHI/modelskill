# ADR-005: Model Result Type Hierarchy

**Status**: Accepted

**Date**: 2021 (evolved incrementally)

## Context

Model output comes in various forms depending on the simulation type and file format: fixed point timeseries (dfs0), moving observations (satellite tracks), regular grids (dfs2, NetCDF), and unstructured meshes (dfsu). Each requires different spatial matching approaches when comparing with observations. Point data can be matched directly, while gridded and mesh data require spatial interpolation.

## Decision

Create a type hierarchy of model result classes: `PointModelResult`, `TrackModelResult`, `GridModelResult`, `DfsuModelResult`, and `DummyModelResult`. Each type implements specialized spatial extraction logic optimized for its data structure.

## Alternatives Considered

**Single ModelResult class with type parameter** - Would require extensive conditional logic and make optimization difficult.

**Separate packages for each model type** - Would fragment the user experience and duplicate comparison logic.

## Consequences

Each type has specialized algorithms for efficient spatial matching. Point and Track types perform direct matches, while Grid and Dfsu extract data via spatial interpolation. File format structure naturally maps to model result types. Different modeling scenarios (coastal, offshore, rivers) use appropriate types. Users can add custom DummyModelResult types for baseline comparisons (e.g., mean, climatology). The hierarchy adds complexity but provides clear separation of concerns and enables performance optimization for each spatial matching strategy.
