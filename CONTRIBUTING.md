# Guidelines for contribution

1. Clone the repository
2. Install [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just) if you don't have them already
3. Install the package with development dependencies:
   ```bash
   uv sync --group dev
   ```
4. Make your changes
5. Run quality checks before committing:
   ```bash
   just check        # Run all checks: lint, typecheck, test, doctest
   # Or run individually:
   just lint         # Check code style with ruff
   just typecheck    # Type check with mypy
   just test         # Run pytest
   just format       # Auto-format code with ruff
   just --list       # Show all available recipes
   ```
6. Make a pull request with a clear summary of the changes

## Running specific tests

```bash
# Run a specific test file
uv run pytest tests/test_comparer.py

# Run a specific test function
uv run pytest tests/test_comparer.py::test_function_name

# Run with coverage
just coverage
```

## Project structure

- Source code is in `src/modelskill/`
- Tests are in `tests/`
- Test data is in `tests/testdata/` (symlinked as `docs/data`)
