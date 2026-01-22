# ADR-006: Dual Plotting Backends

**Status**: Accepted

**Date**: 2021-02

## Context

Users need to visualize comparisons in different contexts: static plots for publications and reports, and interactive plots for exploratory analysis in notebooks. Matplotlib is the standard for static scientific plots, while Plotly provides modern interactive visualizations. Supporting only one backend would limit users' workflow flexibility.

## Decision

Support both matplotlib and plotly as plotting backends, accessed via a `backend` parameter in plot methods. Matplotlib is the default for compatibility and documentation rendering.

## Alternatives Considered

**Matplotlib only** - Would lack interactivity for exploratory analysis in notebooks.

**Plotly only** - Would complicate publication workflows and lose ecosystem familiarity.

**Separate plotting modules** - Would fragment the API and duplicate plot logic.

## Consequences

Users can choose static (matplotlib) or interactive (plotly) visualizations based on their needs. Plotly is an optional dependency for interactive use. Maintaining two backends requires parallel implementation of plot types and increases testing burden. Some plot types may not have feature parity across backends. The `backend` parameter provides a consistent API regardless of the chosen backend.
