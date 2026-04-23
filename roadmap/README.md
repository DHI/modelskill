# ModelSkill Product Roadmap

This roadmap outlines the current and future direction of ModelSkill — a toolkit for evaluating simulation model quality by comparing results against observations.

For questions or feature requests, please open a [GitHub Discussion](https://github.com/DHI/modelskill/discussions).

---


## Delivered


- **[Baseline Model Comparisons](features/baseline-comparisons.md)** — Compare any model against synthetic baselines (mean, persistence) to quantify the added value of a simulation.
- **[Custom Metrics](features/custom-metrics.md)** — Define domain-specific quality metrics that integrate fully into all skill tables and reports.
- **[Spatial and Temporal Skill Aggregation](features/skill-aggregation.md)** — Assess model performance by geographic region, time period, season, or any custom grouping to identify where and when a model performs well or poorly.

## In Development


- **[Network Model Support](features/network-models.md)** — Compare MIKE 1D hydraulic network simulations against observations at network nodes, covering collection systems, water distribution, and river networks.

## Planned


- **[Vertical Profile Assessment](features/vertical-profiles.md)** — Validate 3D models by comparing against depth-varying observations such as temperature and salinity profiles.

## Under Consideration


- **[Automatic Report Generation](features/automatic-reports.md)** — Generate standardised model skill assessment reports in HTML, PDF, or PowerPoint from a single command.
- **[Band-Pass Filtering](features/band-pass-filtering.md)** — Separate model skill assessment into slow dynamics and fast dynamics to understand where a model captures trends versus peaks.
- **[Ensemble and Probabilistic Forecast Support](features/ensemble-support.md)** — Evaluate ensemble model runs using established probabilistic scoring methods alongside standard deterministic metrics.
- **[Forecast Lead-Time Analysis](features/forecast-lead-time.md)** — Assess how model skill degrades with forecast horizon to optimise forecast update frequency and communicate prediction reliability.
- **[Outlier Detection](features/outlier-detection.md)** — Automatically identify suspect observations using model-observation differences to improve data quality and skill assessment reliability.
- **[Rolling Skill Assessment](features/rolling-skill.md)** — Track how model skill evolves over time using moving windows to detect performance trends and seasonal patterns.
- **[Web Application](features/web-app.md)** — Browser-based interface for model skill assessment, accessible to users without Python experience.

## Not Planned

See [features considered out of scope](features/not-planned.md).
