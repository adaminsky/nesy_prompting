.PHONY: create_environment requirements docs docs-serve test \
	test-fastest test-debug-fastest _clean_manual_test manual-test manual-test-debug

## GLOBALS

PROJECT_NAME = unsupervised-nesy
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python


###     DEV COMMANDS

## Set up python interpreter environment
create_environment:
	python -m venv .venv
	@echo ">>> venv created. Activate with:\nsource .venv/bin/activate"

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -r dev-requirements.txt

# ###     DOCS

# docs:
# 	cd docs && mkdocs build

# docs-serve:
# 	cd docs && mkdocs serve

###     TESTS

test: _prep
	pytest -vvv --durations=0

test-fastest: _prep
	pytest -vvv -FFF

test-debug-last:
	pytest --lf --pdb

_clean_manual_test:
	rm -rf manual_test

manual-test: _prep _clean_manual_test
	mkdir -p manual_test
	cd manual_test && python -m ccds ..

manual-test-debug: _prep _clean_manual_test
	mkdir -p manual_test
	cd manual_test && python -m pdb ../ccds/__main__.py ..
