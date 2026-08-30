---
title: "Band-Pass Filtering"
status: "Under Consideration"
category: "Analysis & Metrics"
summary: "Separate model skill assessment into slow dynamics and fast dynamics to understand where a model captures trends versus peaks."
---

## Value Proposition

A model might reproduce long-term trends well but miss short-term peaks, or vice versa. Band-pass filtering allows modellers to decompose the signal into frequency bands and assess skill separately for each — for example, tidal vs. surge components in water level, or diurnal vs. seasonal cycles in temperature.

This provides deeper diagnostic insight into model behaviour, helping modellers understand not just how good a model is overall, but which physical processes it captures well and which need improvement.

## What This Enables

- Decompose observed and modelled time series into frequency bands (e.g., high-frequency peaks, low-frequency trends)
- Assess model skill separately for each band to diagnose where performance is strong or weak
- Compare model versions to see whether improvements target the right frequency range
- Support tidal/surge separation, seasonal/event decomposition, and similar domain-specific analyses

## Current Status

Under consideration. Identified in community discussions as a valuable diagnostic capability. No implementation timeline has been set.
