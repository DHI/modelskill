LIB = src

check: lint typecheck test doctest

build: typecheck test
	uv build

lint:
	uv run ruff check $(LIB)

format:
	uv run ruff format $(LIB)

test:
	uv run pytest --disable-warnings

typecheck:
	uv run mypy $(LIB)/ --config-file pyproject.toml

doctest:
	uv run pytest src/modelskill/metrics.py --doctest-modules

coverage: 
	uv run pytest --cov-report html --cov=$(LIB) tests/

docs: FORCE
	set -e; \
	cd docs; \
	uv run quartodoc build; \
	uv run quarto render; \
	if [ ! -f _site/index.html ]; then \
        echo "Error: index.html not found. Quarto render failed."; \
        exit 1; \
    fi; \
    cd -

clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf dist
	rm -rf docs/_site


FORCE:
