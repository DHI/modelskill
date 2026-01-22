# Guidelines for contribution

1. Clone the repository
2. Install [uv](https://docs.astral.sh/uv/) if you don't have it already
3. Install the package with development dependencies:
   ```bash
   uv sync --group dev
   ```
4. Make your changes
5. Run quality checks before committing:
   ```bash
   make check        # Run all checks: lint, typecheck, test, doctest
   # Or run individually:
   make lint         # Check code style with ruff
   make typecheck    # Type check with mypy
   make test         # Run pytest
   make format       # Auto-format code with ruff
   ```
6. Make a pull request with a clear summary of the changes

## Running specific tests

```bash
# Run a specific test file
uv run pytest tests/test_comparer.py

# Run a specific test function
uv run pytest tests/test_comparer.py::test_function_name

# Run with coverage
make coverage
```

## Project structure

- Source code is in `src/modelskill/`
- Tests are in `tests/`
- Test data is in `tests/testdata/` (symlinked as `docs/data`)
