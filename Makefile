LIB = modelskill

build: typecheck test
	python -m build

lint:
	ruff .

test:
	pytest --disable-warnings

typecheck:
	# TODO remove excludes when done
	mypy $(LIB)/ --exclude $(LIB)/report.py --exclude $(LIB)/settings.py --exclude $(LIB)/connection.py

coverage: 
	pytest --cov-report html --cov=$(LIB) tests/

docs: FORCE
	cd docs; make html ;cd -

FORCE:
