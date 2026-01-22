# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for ModelSkill. ADRs document significant architectural choices made in the project, explaining the context, decision, alternatives considered, and consequences.

## Purpose

ADRs help new developers (and future maintainers) understand why the codebase is structured the way it is. They capture the reasoning behind key design decisions at the time they were made.

## Format

Each ADR follows this structure:

- **Status**: Draft, Accepted, Superseded
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
- [ADR-010](010-optional-domain-dependencies.md) - Optional dependencies for domain-specific model types (Draft)

## Contributing

When making significant architectural changes, please:

1. Create a new ADR in Draft status
2. Discuss with the team
3. Update to Accepted status once implemented
4. Update this index with a link to the new ADR

### Superseding an Existing ADR

When a new decision replaces an old one:

1. Create the new ADR following the normal process
2. In the new ADR, include a note in the Context section mentioning which ADR it supersedes (e.g., "This decision supersedes [ADR-001](001-previous-decision.md)")
3. Update the Status field of the old ADR from "Accepted" to "Superseded"
4. Do NOT modify the body of the old ADR (Context, Decision, Alternatives, Consequences) - it remains as an immutable historical record
5. Both ADRs remain in the repository to preserve the full decision history
