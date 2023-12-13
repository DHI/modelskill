LIB = modelskill

check: lint typecheck test doctest

build: typecheck test
	python -m build

lint:
	ruff .

test:
	pytest --disable-warnings

typecheck:
	mypy $(LIB)/ --config-file pyproject.toml

doctest:
	pytest ./modelskill/metrics.py --doctest-modules

coverage: 
	pytest --cov-report html --cov=$(LIB) tests/

docs: FORCE
	mkdocs build

FORCE:
