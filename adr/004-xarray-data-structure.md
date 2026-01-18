# ADR-004: xarray as Internal Data Structure

**Status**: Accepted

**Date**: 2021-04

## Context

Comparers needed to store matched observation and model data with associated metadata (timestamps, spatial coordinates, variable names, units, observation types). Initially, pandas DataFrames were used but managing metadata separately from data proved cumbersome. A critical requirement was the ability to persist Comparer objects to disk and resume work later.

## Decision

Adopt xarray Datasets and DataArrays as the internal data structure for storing matched comparison data within Comparer objects.

## Alternatives Considered

**Continue with pandas DataFrames** - Limited metadata handling and no straightforward persistence mechanism that preserves all attributes.

**Custom data container with pickle** - Would require custom serialization logic and lack interoperability with standard formats.

## Consequences

Integrated metadata and attribute management. Persistence to disk via NetCDF is built-in, allowing users to save Comparers and resume analysis later. xarray provides labeled dimensions, coordinate handling, and ecosystem compatibility. Users access the underlying xarray Dataset via `comparer.data`. The package can leverage xarray's capabilities like dask integration for large datasets. Adds xarray as a core dependency.
