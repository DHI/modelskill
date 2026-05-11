---
title: "Forecast Lead-Time Analysis"
status: "Under Consideration"
category: "Analysis & Metrics"
summary: "Assess how model skill degrades with forecast horizon to optimise forecast update frequency and communicate prediction reliability."
---

## Value Proposition

Operational forecasting systems produce predictions at varying lead times — a 6-hour forecast is typically more accurate than a 48-hour forecast. Understanding exactly how skill degrades with lead time is essential for deciding how often to update forecasts and for communicating to end users how far ahead they can trust predictions.

This capability would enable forecast managers to make data-driven decisions about operational scheduling and to set appropriate confidence levels for different forecast horizons.

## What This Enables

- Evaluate model skill as a function of forecast lead time (e.g., skill at T+6h, T+12h, T+24h, T+48h)
- Identify the forecast horizon beyond which predictions become unreliable
- Compare lead-time degradation across different model versions or configurations
- Support decisions about forecast update frequency and operational scheduling
- Produce lead-time skill curves for inclusion in forecast service documentation

## Current Status

Under consideration. The underlying architecture can accommodate lead-time analysis, and early design thinking has been done. No implementation timeline has been set.
