set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

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
[unix]
docs:
    cd docs && uv run quartodoc build && uv run quarto render
    test -f docs/_site/index.html || { echo "Error: index.html not found."; exit 1; }

[windows]
docs:
    cd docs; uv run quartodoc build; uv run quarto render
    if (!(Test-Path docs/_site/index.html)) { Write-Error "Error: index.html not found."; exit 1 }

# Build docs without executing Python (fast; produces a link-checkable _site/)
[unix]
docs-fast:
    cd docs && uv run quartodoc build && uv run quarto render --no-execute

[windows]
docs-fast:
    cd docs; uv run quartodoc build; uv run quarto render --no-execute

# Check links in .qmd sources (offline, no render needed)
linkcheck:
    lychee --offline --include-fragments 'docs/**/*.qmd'

# Check links in rendered site (catches quartodoc cross-refs and anchors)
linkcheck-site: docs-fast
    lychee --offline --include-fragments 'docs/_site/**/*.html'

# Check external URLs in .qmd sources (hits the network)
linkcheck-online:
    lychee --include-fragments 'docs/**/*.qmd'

# Clean build artifacts
[unix]
clean:
    rm -rf .pytest_cache .mypy_cache .coverage dist docs/_site

[windows]
clean:
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .pytest_cache, .mypy_cache, .coverage, dist, docs/_site
