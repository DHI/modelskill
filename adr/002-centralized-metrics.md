# ADR-002: Centralized Metrics Module

**Status**: Accepted

**Date**: 2021-02

## Context

Model skill assessment requires various statistical metrics (bias, RMSE, correlation, skill scores) to quantify agreement between model results and observations. These metrics needed to be applied consistently, extended easily, and available for users to call directly.

## Decision

Create a centralized `metrics.py` module containing all statistical metric functions as standalone functions operating on numpy arrays.

## Alternatives Considered

**Embed metrics as methods in Comparer classes** - Would couple metrics tightly to comparison objects, reducing reusability.

**Distribute metrics across relevant modules** - Would make metrics harder to discover and maintain.

## Consequences

Metrics are reusable outside of Comparer objects, easy to discover in one location, and straightforward to test independently. Custom metrics can be registered centrally. The metrics module has grown large and currently has mypy type checking disabled. All metrics must work with numpy array inputs.
