# Run all checks: lint, typecheck, test, doctest
check: lint typecheck test doctest

# Build package (after typecheck and test)
build: typecheck test
    uv build

# Lint with ruff
lint:
    uv run ruff check src

# Auto-fix formatting
format:
    uv run ruff format src

# Run tests
test:
    uv run pytest --disable-warnings

# Type check with mypy
typecheck:
    uv run mypy src/ --config-file pyproject.toml

# Run doctests in metrics.py
doctest:
    uv run pytest src/modelskill/metrics.py --doctest-modules

# Generate HTML coverage report
coverage:
    uv run pytest --cov-report html --cov=src tests/

# Build documentation
docs:
    cd docs && uv run quartodoc build && uv run quarto render
    test -f docs/_site/index.html || { echo "Error: index.html not found."; exit 1; }

# Clean build artifacts
clean:
    rm -rf .pytest_cache .mypy_cache .coverage dist docs/_site
