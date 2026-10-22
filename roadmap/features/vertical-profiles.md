---
title: "Vertical Profile Assessment"
status: "In Development"
category: "Domain Expansion"
summary: "Validate 3D models by comparing against depth-varying observations such as temperature and salinity profiles."
---

## Value Proposition

Three-dimensional hydrodynamic models produce results that vary with depth — temperature stratification, salinity gradients, and current profiles are critical outputs for environmental assessments, aquaculture siting, and water quality management. Today, validating these depth-varying outputs requires significant manual effort to align model layers with observation depths.

Supporting vertical profiles natively will allow modellers to validate 3D model performance with the same ease as surface-level comparisons, providing confidence in the full water column representation.

## What This Enables

- Compare model output at multiple depths against profiling instruments (CTD casts, ADCP profiles)
- Assess skill as a function of depth to identify where in the water column the model performs best
- Visualise observed vs modelled profiles side by side
- Aggregate skill across depth bins for summary reporting
- Support different vertical grid types used in 3D models

## Current Status

In active development, currently at an alpha stage. `VerticalObservation` and
`VerticalModelResult` exist, along with depth-resolved skill (`SkillProfile`) and profile
plotting. The surface is not yet stable and is subject to change before release.
