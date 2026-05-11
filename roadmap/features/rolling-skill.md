---
title: "Rolling Skill Assessment"
status: "Under Consideration"
category: "Analysis & Metrics"
summary: "Track how model skill evolves over time using moving windows to detect performance trends and seasonal patterns."
---

## Value Proposition

A single skill score for an entire simulation period can mask important temporal patterns. Model performance may drift over time due to changing boundary conditions, seasonal effects, or data assimilation quality. Rolling skill assessment reveals these trends, enabling operational teams to detect when a model starts underperforming and take corrective action.

This is particularly valuable for operational forecasting systems where continuous performance monitoring is essential for maintaining service quality.

## What This Enables

- Compute skill metrics over a moving time window to produce a skill time series
- Detect performance degradation or improvement trends over the simulation period
- Identify seasonal patterns in model skill (e.g., consistently lower skill in winter)
- Monitor operational model performance in near-real-time dashboards
- Compare rolling skill between model versions to verify that updates improve performance consistently, not just on average

## Current Status

Under consideration. Identified in community discussions as a valuable operational monitoring capability. No implementation timeline has been set.
