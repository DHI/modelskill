LIB = modelskill

check: lint typecheck test doctest

build: typecheck test
	python -m build

lint:
	ruff check $(LIB)

format:
	ruff format $(LIB)

test:
	pytest --disable-warnings

typecheck:
	mypy $(LIB)/ --config-file pyproject.toml

doctest:
	pytest ./modelskill/metrics.py --doctest-modules

coverage: 
	pytest --cov-report html --cov=$(LIB) tests/

docs: FORCE
	set -e; \
	cd docs; \
	quartodoc build; \
	quarto render; \
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
