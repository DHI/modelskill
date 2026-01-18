# ADR-001: mikeio as Core Dependency

**Status**: Accepted

**Date**: 2021-01 (Initial commit)

## Context

ModelSkill was created to evaluate model skill by comparing MIKE FM simulation results with observations. MIKE file formats (dfs0, dfs2, dfsu) are complex binary formats with specific metadata structures, unstructured mesh geometries, and temporal indexing that require specialized handling.

At the time ModelSkill was created, `mikeio` already existed as DHI's Python package for reading and writing MIKE files, providing a Pythonic interface to MIKE formats.

## Decision

Use `mikeio` as a core dependency for all MIKE file format I/O operations.

## Alternatives Considered

None. This was the obvious choice given:
- mikeio already provided the needed functionality
- ModelSkill and mikeio are both DHI packages with aligned maintenance
- Separation of concerns: file I/O belongs in mikeio, comparison/skill belongs in ModelSkill
- Avoiding code duplication across the DHI Python ecosystem

## Consequences

**Positive:**
- **Separation of concerns**: ModelSkill focuses on comparison and skill assessment logic
- **Reusability**: Leverages existing, well-tested file I/O functionality
- **Maintenance**: MIKE format changes are handled by mikeio maintainers
- **Ecosystem integration**: Part of coherent DHI Python toolchain
- **Feature access**: Automatic access to all MIKE file format features

**Negative:**
- Introduces external dependency with version requirements
- ModelSkill's MIKE file capabilities are bounded by mikeio's API

**Note**: The package was initially named "fmskill" reflecting its focus on MIKE FM models, later renamed to "modelskill" as support for other model types and file formats was added.
