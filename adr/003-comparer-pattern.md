# ADR-003: Comparer and ComparerCollection Pattern

**Status**: Accepted

**Date**: 2021-03 (evolved from 2021-02)

## Context

After matching observations with model results, users need to analyze comparison data, calculate skill metrics, and create visualizations. Initial implementation used an inheritance hierarchy with `BaseComparer` and specialized subclasses (`PointComparer`, `TrackComparer`) for different observation types. This pattern proved rigid when handling multiple observations.

## Decision

Adopt a composition pattern with `Comparer` for single observation comparisons and `ComparerCollection` as a container of multiple Comparers. Both provide similar interfaces (`.skill()`, `.plot`) adapted to their scope.

## Alternatives Considered

**Continue with inheritance hierarchy** - Required complex type-specific subclasses and made multi-observation aggregation difficult.

**Single unified class** - Would require extensive conditional logic to handle single vs multiple observation scenarios.

## Consequences

Clean separation between single and multi-observation workflows. Composition over inheritance makes the code more flexible and easier to extend. ComparerCollection provides dictionary-like access to individual Comparers. Some API duplication exists between Comparer and ComparerCollection. The inheritance-based classes (BaseComparer, PointComparer, TrackComparer) were eventually removed (2023).
