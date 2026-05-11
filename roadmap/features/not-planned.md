---
title: "Not Planned"
status: "Not Planned"
category: "Out of Scope"
summary: "Features that have been considered and determined to be outside ModelSkill's scope."
---

## Extreme Value Analysis

Statistical analysis of extreme events (return periods, GEV fitting, peaks-over-threshold) is a specialised domain with dedicated statistical tools designed for this purpose. ModelSkill focuses on routine skill assessment of model outputs against observations, not on the statistical characterisation of extremes. Users needing extreme value analysis should apply those methods to ModelSkill's matched datasets using established statistical packages.

## Deterministic Wave Analysis

Wave crossing analysis, spectral decomposition, and other signal-processing techniques for wave data are best served by dedicated wave analysis tooling. ModelSkill evaluates how well a model reproduces observed values — it does not perform signal-level analysis of the data itself.

## Project-Specific File Formats

ModelSkill supports widely used data formats: NetCDF, CSV, tabular data, and MIKE file formats (dfs0, dfs2, dfsu). Supporting niche or project-specific file formats would serve only a small number of users while requiring ongoing maintenance. Users with non-standard formats should convert their data to a supported format before loading into ModelSkill.

## Timezone-Aware Timestamps

Model comparison workflows assume all data is in a consistent time reference (typically UTC). Adding timezone handling throughout the matching and comparison process would introduce significant complexity without clear benefit for skill assessment, since the prerequisite for meaningful comparison is that observation and model data are already on the same time reference.
