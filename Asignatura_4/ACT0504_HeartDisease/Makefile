VENV=.venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

.PHONY: venv install run clean

venv:
	python3 -m venv $(VENV)

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	PYTHONHASHSEED=0 $(PY) main.py

clean:
	rm -rf .pytest_cache .mypy_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +

