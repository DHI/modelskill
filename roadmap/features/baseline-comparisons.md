---
title: "Baseline Model Comparisons"
status: "Delivered"
category: "Analysis & Metrics"
summary: "Compare any model against synthetic baselines (mean, persistence) to quantify the added value of a simulation."
---

## Value Proposition

When presenting model results to stakeholders, a common question is: "How much better is this model than a simple guess?" Baseline comparisons answer that question by letting you evaluate your model against trivial reference models — such as using the observed mean or simply repeating the last known measurement.

This turns model validation from "the RMSE is 0.3 m" (which means little on its own) into "our model reduces error by 40% compared to using the historical average" — a statement that clearly communicates the value of the modelling investment.

## What This Enables

- Compare any simulation against a **mean baseline** (constant prediction equal to the observed average)
- Compare against a **persistence baseline** (repeat the last observation forward)
- Define **custom baselines** such as climatological averages or seasonal means
- Compute standard skill scores (e.g., Nash-Sutcliffe Efficiency) that inherently measure improvement over a reference
- Present model value in terms stakeholders can immediately understand

## Current Status

Delivered in ModelSkill v1.3. Users can create synthetic baseline references and include them alongside real model results in skill assessments.
