LIB = modelskill

check: lint typecheck test

build: typecheck test
	python -m build

lint:
	ruff .

test:
	pytest --disable-warnings

typecheck:
	mypy $(LIB)/ --config-file pyproject.toml

coverage: 
	pytest --cov-report html --cov=$(LIB) tests/

docs: FORCE
	cd docs; make html ;cd -

FORCE:
