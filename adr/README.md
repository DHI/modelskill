# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for ModelSkill. ADRs document significant architectural choices made in the project, explaining the context, decision, alternatives considered, and consequences.

## Purpose

ADRs help new developers (and future maintainers) understand why the codebase is structured the way it is. They capture the reasoning behind key design decisions at the time they were made.

## Format

Each ADR follows this structure:

- **Status**: Draft, Accepted, Superseded, Deprecated
- **Date**: When the decision was made (approximate for historical ADRs)
- **Context**: What problem or requirement led to this decision?
- **Decision**: What did we decide to do?
- **Alternatives Considered**: What other options were evaluated?
- **Consequences**: What are the trade-offs and implications?

## Index

- [ADR-001](001-mikeio-dependency.md) - mikeio as core dependency
- [ADR-002](002-centralized-metrics.md) - Centralized metrics module
- [ADR-003](003-comparer-pattern.md) - Comparer and ComparerCollection pattern
- [ADR-004](004-xarray-data-structure.md) - xarray as internal data structure
- [ADR-005](005-model-result-hierarchy.md) - Model result type hierarchy
- [ADR-006](006-dual-plotting-backends.md) - Dual plotting backends (matplotlib and plotly)
- [ADR-007](007-four-step-workflow.md) - Four-step workflow pattern
- [ADR-008](008-options-styling-system.md) - Options and styling system
- [ADR-009](009-factory-pattern.md) - Factory pattern for type detection

## Contributing

When making significant architectural changes, consider documenting them as a new ADR. Start with status "Draft" and update to "Accepted" once the change is implemented and reviewed.
