# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ModelSkill is a Python package for evaluating model skill by comparing simulation results with observations. It's primarily used for MIKE models but supports other models as well. The package handles various types of spatial and temporal data (point observations, tracks, gridded fields, DFSU files) and provides comprehensive statistical analysis and visualization.

## Development Commands

### Package Management
This project uses `uv` for dependency management. Install dependencies with:
```bash
uv sync --group dev     # Install with dev dependencies
uv sync --group test    # Install with test dependencies
```

### Testing
```bash
make test               # Run all tests (ignores notebooks)
pytest                  # Direct pytest invocation
pytest tests/test_comparer.py::test_name  # Run specific test
pytest --disable-warnings  # Run without warnings (default in Makefile)
```

### Code Quality
```bash
make check              # Run all checks: lint, typecheck, test, doctest
make lint               # Lint with ruff
make format             # Format with ruff
make typecheck          # Type check with mypy
make doctest            # Run doctests in metrics.py
make coverage           # Generate HTML coverage report
```

### Building
```bash
make build              # Run typecheck and test, then build package with uv build
uv build                # Build wheel and sdist
```

### Documentation
```bash
make docs               # Build documentation with quartodoc and quarto
                        # Located in docs/_site after building
```

## Coding Conventions

### Docstrings
- All docstrings use **NumPy format** (not Google or reStructuredText style)
- Include sections: Parameters, Returns, Raises, Examples, See Also, Notes as appropriate
- Example:
  ```python
  def function_name(param1, param2):
      """Short description.

      Longer description if needed.

      Parameters
      ----------
      param1 : type
          Description of param1
      param2 : type
          Description of param2

      Returns
      -------
      type
          Description of return value
      """
  ```

## Architecture

### Core Workflow (4-Step Pattern)
The package follows a consistent 4-step workflow that users should follow:

1. **Define ModelResults** - Load/create model output data
2. **Define Observations** - Load/create observation data
3. **Match** - Spatially and temporally match observations with model results
4. **Compare** - Analyze and visualize using Comparer/ComparerCollection objects

### Key Components

#### Model Results (`src/modelskill/model/`)
Model results represent simulation output. Types inherit from a base class:
- `PointModelResult` - Fixed point timeseries (dfs0, nc, DataFrame)
- `TrackModelResult` - Moving point timeseries (dfs0, nc, DataFrame)
- `GridModelResult` - Regular gridded fields (dfs2, nc, xarray Dataset) - extractable via spatial interpolation
- `DfsuModelResult` - Unstructured mesh fields (dfsu files) - extractable via spatial interpolation
- `DummyModelResult` - Synthetic baseline for skill comparison (e.g., mean, climatology)

Use `model_result()` factory function to auto-detect type from input data.

#### Observations (`src/modelskill/obs.py`)
Observations represent measured data:
- `PointObservation` - Fixed location timeseries
- `TrackObservation` - Moving location timeseries (e.g., satellite altimetry)

Use `observation()` factory function to auto-detect type from input data.

#### Matching (`src/modelskill/matching.py`)
The `match()` function aligns observations with model results in space and time:
- Spatial matching: extracts model data at observation locations (for Grid/Dfsu)
- Temporal matching: aligns timestamps within tolerance
- Returns `Comparer` (single obs) or `ComparerCollection` (multiple obs)

Can also use `from_matched()` when data is pre-aligned.

#### Comparison (`src/modelskill/comparison/`)
The core analysis objects after matching:
- `Comparer` - Single observation vs model result(s) comparison
  - Contains matched xarray Dataset with observation and model data
  - Has `.plot` attribute (ComparerPlotter) for visualization
  - Provides `.skill()` method returning SkillTable
  - Supports filtering, selecting, and aggregation
- `ComparerCollection` - Multiple Comparers for multi-observation analysis
  - Dictionary-like access to individual Comparers
  - Has `.plot` attribute (ComparerCollectionPlotter) for multi-obs plots
  - Aggregated skill across observations

#### Metrics and Skill (`src/modelskill/metrics.py`, `src/modelskill/skill.py`)
- `metrics.py` - All statistical metrics (bias, rmse, r2, skill scores, etc.)
  - Supports both scalar and directional (circular) metrics
  - Add custom metrics by registering functions
- `SkillTable` - DataFrame-like container for skill assessment results
  - Multi-level indexing support (observation, model, variable, etc.)
  - Styled HTML output for reports
  - Plotting capabilities for metric visualization

#### Plotting (`src/modelskill/plotting/`)
Visualization modules:
- `_scatter.py` - Scatter plots for model vs observation
- `_spatial_overview.py` - Maps showing observation locations
- `_temporal_coverage.py` - Timeline plots of data availability
- `_taylor_diagram.py` - Taylor diagrams for skill visualization
- `_wind_rose.py` - Directional data visualization

Plots support both matplotlib (static) and plotly (interactive) backends.

#### Configuration (`src/modelskill/configuration.py`)
`from_config()` allows workflow definition via YAML/dict for reproducibility.

### Data Structure Notes

- Internal data storage uses xarray Datasets with standardized coordinate/variable names
- Time coordinates use pandas datetime64
- Spatial coordinates: `x`, `y` (and `z` when applicable)
- Reserved names in `_RESERVED_NAMES` should not be used for model/observation names
- The `Quantity` class handles physical quantities with units and validation

### Testing Structure (`tests/`)

- `test_*.py` - Main unit tests
- `model/` - Model result loading tests
- `observation/` - Observation loading tests
- `integration/` - End-to-end workflow tests
- `plot/` - Visualization tests
- `regression/` - Regression test data
- `testdata/` - Sample data files (symlinked from docs/data)

## Important Notes

- The package depends on MIKE IO (`mikeio`) for reading MIKE file formats (dfs0, dfs2, dfsu)
- Type checking with mypy ignores errors in `metrics.py` module (see pyproject.toml)
- Notebooks in `notebooks/` are excluded from pytest by default (pytest.ini)
- Documentation uses Quarto with quartodoc for API reference generation
- Python 3.10+ required; supports through 3.13
