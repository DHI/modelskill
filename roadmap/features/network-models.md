---
title: "Network Model Support"
status: "In Development"
category: "Domain Expansion"
summary: "Compare MIKE 1D hydraulic network simulations against observations at network nodes, covering collection systems, water distribution, and river networks."
---

## Value Proposition

Hydraulic network models — for urban drainage, water distribution, and river systems — are a major part of DHI's modelling portfolio. Currently, validating these models against observed data requires custom scripts and manual workflows. Adding native support for MIKE 1D results brings the same structured, reproducible validation workflow to network modellers that already exists for coastal and marine models.

This reduces the effort required to produce quality-assured model deliverables and ensures consistent validation standards across all model types within an organisation.

## What This Enables


- Load MIKE 1D, MIKE 11 and EPANET simulation results as model results
- Match network model outputs against point observations at specific nodes or reaches
- Apply the full suite of ModelSkill metrics and visualisations to network model validation
- Compare multiple network model scenarios side by side
- Produce standardised skill assessments for urban drainage, water supply, and river modelling projects

## Current Status

In active development. MIKE 1D, MIKE 11 and EPANET result files can be read today. Integration with ModelSkill's validation workflow is underway.

MOUSE and Water Hammer results are not read yet: no shareable result file exists for either format, so support cannot be verified. SWMM results are not read yet: the reach connectivity lives in the companion '.inp' input file, which modelskill does not read yet.

The layer that reads those files and builds the network lives in mikeio1d, where the formats, the fixtures and a first graph conversion already were (ADR-013). ModelSkill keeps the model result, the observations and the matching, and requires `mikeio1d[network]`. The release is coordinated: this feature ships once mikeio1d has released the module it depends on.

