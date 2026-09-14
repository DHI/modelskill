set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Run all checks: lint, private-access, typecheck, test, doctest
check: lint private-access typecheck test doctest

# Build package (after typecheck and test)
build: typecheck test
    uv build

# Lint with ruff
lint:
    uv run ruff check src tools

# Auto-fix formatting
format:
    uv run ruff format src tools

# Run tests
test:
    uv run pytest --disable-warnings

# Report private attribute access on third-party objects
private-access:
    uv run python tools/check_third_party_private_access.py

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
[unix]
docs:
    cd docs && uv run quartodoc build && uv run quarto render
    test -f docs/_site/index.html || { echo "Error: index.html not found."; exit 1; }

[windows]
docs:
    cd docs; uv run quartodoc build; uv run quarto render
    if (!(Test-Path docs/_site/index.html)) { Write-Error "Error: index.html not found."; exit 1 }

# Clean build artifacts
[unix]
clean:
    rm -rf .pytest_cache .mypy_cache .coverage dist docs/_site

[windows]
clean:
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .pytest_cache, .mypy_cache, .coverage, dist, docs/_site
