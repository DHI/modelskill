# ADR-010: Optional Dependencies for Domain-Specific Model Types

**Status**: Draft

**Date**: 2026-01

## Context

Support for MIKE 1D network model results requires mikeio1d, which depends on pythonnet for .NET interop. This adds installation complexity, platform-specific requirements, and package bloat for the majority of modelskill users who work with 2D/3D models. Network models serve a distinct user community with specialized needs. Future domain-specific extensions (e.g., SWMM, TUFLOW) may have similar characteristics.

## Decision

Make domain-specific model type dependencies optional rather than required. Users install only the dependencies needed for their model types.

## Alternatives Considered

**Required dependency (like mikeio)** - Would burden all users with .NET installation complexity and package bloat for functionality most won't use.

**Separate package (modelskill-network)** - Creates fragmentation, complicates documentation, and requires duplicate infrastructure. Users must discover and install extensions separately.

**Auto-detection via lazy import** - Import mikeio1d only when NetworkModelResult is used. Provides convenience but gives unclear installation errors and makes dependency management implicit.

**Optional extras (pip install modelskill[network])** - Standard Python pattern for optional features. Explicit dependencies, clear installation path, single package.

## Implementation Options

### Option A: Extras pattern (recommended)
```python
# pyproject.toml
[dependency-groups]
network = ["mikeio1d"]
```
Installation: `pip install modelskill[network]` or `uv add modelskill --extra network`

### Option B: Auto-detection
Import mikeio1d conditionally; raise helpful error if missing when NetworkModelResult is used.
Installation: `pip install modelskill mikeio1d`

### Option C: Separate package
Create `modelskill-network` as extension package.
Installation: `pip install modelskill modelskill-network`

## Consequences

**Positive:**
- Core package remains lightweight for majority use case
- .NET complexity isolated to users who need network models
- Establishes extensibility pattern for future domain-specific types
- Explicit opt-in clarifies what users are installing

**Negative:**
- Documentation must clearly explain optional features and installation
- Testing matrix expands (with/without optional dependencies)
- NetworkModelResult code must handle missing dependency gracefully
- Some users may not discover network model support

**Open Questions:**
- Should `modelskill[all]` install all optional model types?
- How to handle version constraints for optional dependencies?
- Should optional dependencies be tested in CI for every commit or separately?

## Status Notes

This ADR is in Draft status pending:
- Selection of implementation approach (extras vs auto-detection vs separate package)
- Team review and feedback
- Implementation and testing
