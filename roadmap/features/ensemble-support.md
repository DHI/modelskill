---
title: "Ensemble and Probabilistic Forecast Support"
status: "Under Consideration"
category: "Analysis & Metrics"
summary: "Evaluate ensemble model runs using established probabilistic scoring methods alongside standard deterministic metrics."
---

## Value Proposition

Ensemble forecasting — running multiple model realisations to capture uncertainty — is increasingly common in operational hydrology, meteorology, and coastal forecasting. Standard deterministic metrics (RMSE, bias) cannot properly evaluate probabilistic predictions. Proper scoring rules like the Continuous Ranked Probability Score (CRPS) are needed to assess whether the ensemble spread accurately reflects forecast uncertainty.

Supporting ensemble evaluation would position ModelSkill as a complete validation toolkit for both deterministic and probabilistic forecasting workflows.

## What This Enables

- Load and compare ensemble model runs (multiple realisations) against observations
- Compute probabilistic metrics: CRPS, rank histograms, reliability diagrams, Brier score
- Assess ensemble spread calibration — is the ensemble over-confident or under-confident?
- Compare deterministic and ensemble forecasts using appropriate metrics for each
- Produce ensemble-specific visualisations (spaghetti plots, fan charts, probability exceedance curves)

## Current Status

Under consideration. This feature would require expanding ModelSkill's internal structure to handle multiple ensemble members and adding probabilistic scoring metrics. No implementation timeline has been set.
