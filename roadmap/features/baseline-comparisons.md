---
title: "Baseline Model Comparisons"
status: "Delivered"
category: "Analysis & Metrics"
summary: "Compare any model against synthetic baselines (mean, constant) to quantify the added value of a simulation."
---

## Value Proposition

When presenting model results to stakeholders, a common question is: "How much better is this model than a simple guess?" Baseline comparisons answer that question by letting you evaluate your model against trivial reference models — such as using the observed mean or a fixed constant value.

This turns model validation from "the RMSE is 0.3 m" (which means little on its own) into "our model reduces error by 40% compared to using the historical average" — a statement that clearly communicates the value of the modelling investment.

## What This Enables

- Compare any simulation against a **mean baseline** (constant prediction equal to the observed average)
- Compare against a **constant baseline** (a fixed value chosen by the user)
- Compute standard skill scores (e.g., Nash-Sutcliffe Efficiency) that inherently measure improvement over a reference
- Present model value in terms stakeholders can immediately understand

## Not Included

Two further baselines common in forecast verification are **not** available as built-in
strategies: a **persistence baseline** (repeat the last observation forward by a fixed lag) and a
**climatology baseline** (day-of-year or monthly means of the observations). Both can be
constructed by hand today by building the series with pandas and passing it as a
`PointModelResult`. Note that a persistence baseline is lead-time dependent — the T+24h forecast
uses the observation 24h earlier — so a persistence reference across forecast horizons depends on
[Forecast Lead-Time Analysis](forecast-lead-time.md).

## Current Status

The `mean` and `constant` baseline strategies are delivered, available since v1.1 via
`DummyModelResult`. Baselines can be included alongside real model results in skill assessments.
