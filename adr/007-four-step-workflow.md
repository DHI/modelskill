# ADR-007: Four-Step Workflow Pattern

**Status**: Accepted

**Date**: 2021-2023 (evolved organically)

## Context

As the package developed, different usage patterns emerged. Early implementations mixed data loading, spatial matching, and skill calculation in various ways. Through user feedback and API refinement, a consistent workflow pattern crystallized that balanced clarity, flexibility, and maintainability.

## Decision

Standardize on a four-step workflow: (1) Define ModelResults, (2) Define Observations, (3) Match spatially and temporally, (4) Compare and analyze. Each step is explicit and produces objects for the next step.

## Alternatives Considered

**Single-function convenience API** - Would hide complexity but reduce flexibility in matching parameters and intermediate inspection.

**Two-step workflow (load + compare)** - Would combine matching with comparison, making it harder to debug spatial/temporal alignment issues.

## Consequences

Explicit workflow makes the process transparent and educational for users. Each step provides control over parameters (spatial tolerance, temporal matching, etc.). Users can inspect intermediate results (matched data before comparison). Separation of concerns makes testing easier. The pattern is consistent across all model and observation types. Requires more code than a single convenience function but provides better understanding of the comparison process. Documentation and examples consistently teach this workflow.
