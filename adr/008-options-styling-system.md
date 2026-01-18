# ADR-008: Options and Styling System

**Status**: Accepted

**Date**: 2023-01

## Context

Users needed consistent visual styling across plots and skill tables for different output contexts. Researchers required publication-quality defaults while DHI consultants needed plots matching MOOD (Met-Ocean On Demand) branding for client reports. Specifying styling parameters repeatedly for each plot was tedious and error-prone.

## Decision

Implement a global options and styling system that can be configured via YAML files or programmatically. Provide built-in styles including a MOOD style matching DHI's visual identity. Users can load styles with `modelskill.load_style()` or create custom style configurations.

## Alternatives Considered

**Plot-level styling only** - Would require repetitive parameter passing and make consistent styling across a project difficult.

**Hard-coded DHI branding** - Would not serve researchers or users needing custom styling.

**Theme classes instead of YAML** - Would be less accessible to non-programmers and harder to share configurations.

## Consequences

Users can apply consistent styling globally without repetitive parameters. MOOD style enables DHI-branded outputs for reports. Custom styles support different publication requirements. Configuration via YAML files is shareable and version-controllable. The system covers both plot appearance and skill table styling. Global state can lead to unexpected behavior if styles are changed mid-analysis. The styling system adds complexity to the plotting modules.
