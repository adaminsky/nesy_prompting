.PHONY: create_environment requirements clean data

## GLOBALS

PYTHON_INTERPRETER = python


###     DEV COMMANDS

## Set up python interpreter environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv .venv
	@echo ">>> venv created. Activate with:\nsource .venv/bin/activate"

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


data: requirements
	$(PYTHON_INTERPRETER) src/dataset.py


# ###     DOCS
# NOTE: Commenting out docs for now since we currently have no documentation
# docs:
# 	cd docs && mkdocs build

# docs-serve:
# 	cd docs && mkdocs serve


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
