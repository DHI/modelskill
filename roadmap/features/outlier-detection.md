---
title: "Outlier Detection"
status: "Under Consideration"
category: "Analysis & Metrics"
summary: "Automatically identify suspect observations using model-observation differences to improve data quality and skill assessment reliability."
---

## Value Proposition

Observation data often contains errors — sensor malfunctions, biofouling, transmission glitches — that can distort skill assessments. Currently, identifying and removing these outliers is a manual process. Using the model-observation difference as a diagnostic signal can flag suspect data points systematically, improving both the reliability of skill metrics and the quality of observation datasets.

This benefits both model validation (more accurate skill scores) and data management (systematic quality control of observation networks).

## What This Enables

- Flag observations that deviate significantly from model predictions as potential outliers
- Review flagged points before deciding whether to include or exclude them from skill assessment
- Apply configurable thresholds based on standard deviations, percentiles, or domain-specific criteria
- Improve consistency and defensibility of skill assessments by documenting data quality decisions
- Feed back data quality insights to observation network operators

## Current Status

Under consideration. Identified in community discussions as a practical need for working with real-world observation data. No implementation timeline has been set.
