# ADR-009: Factory Pattern for Type Detection

**Status**: Accepted

**Date**: 2023-12 (v1.0.b1)

## Context

Users needed to import and instantiate the correct model result or observation class based on their data. With five model result types (Point, Track, Grid, Dfsu, Dummy) and two observation types (Point, Track), users had to understand the type hierarchy and import the appropriate class. This created friction in the user experience and required knowledge of internal class structure.

## Decision

Provide factory functions `model_result()` and `observation()` that automatically detect the appropriate type from the input data structure and return the correct class instance. Users interact with a single entry point regardless of their data type.

## Alternatives Considered

**Require explicit class import** - Would burden users with understanding the type hierarchy and remembering which class to use.

**Single class with type parameter** - Would still require users to specify the type and understand the distinctions.

## Consequences

Simplified API with single entry point for creating model results and observations. Users don't need to know internal class names or hierarchy. Type detection logic is centralized and testable. Reduces boilerplate in user code. Type inference happens automatically based on data structure (file extension, presence of spatial coordinates, etc.). May hide type-specific options from users who need fine-grained control. Type detection can fail for ambiguous inputs, requiring fallback to explicit class instantiation.
