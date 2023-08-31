LIB = modelskill

build: typecheck test
	python -m build

lint:
	ruff .

test:
	pytest --disable-warnings

typecheck:
	mypy $(LIB)/

coverage: 
	pytest --cov-report html --cov=$(LIB) tests/

docs: FORCE
	cd docs; make html ;cd -

FORCE:
