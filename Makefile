.PHONY: help venv ingest eval test clean

help:
	@echo "Targets:"
	@echo "  venv    - create local python venv and install requirements"
	@echo "  ingest  - convert data/raw/*.csv -> data/parquet/ (see src/etl/ingest.py)"
	@echo "  eval    - honest out-of-fold ROC/PR-AUC report (src/eval/run_auc.py)"
	@echo "  test    - run pytest"
	@echo "  clean   - remove generated parquet + caches"

venv:
	python3 -m venv venv
	./venv/bin/pip install -U pip
	test -f requirements.txt && ./venv/bin/pip install -r requirements.txt || true

ingest:
	./venv/bin/python src/etl/ingest.py

eval:
	./venv/bin/python src/eval/run_auc.py

test:
	./venv/bin/python -m pytest -q

clean:
	rm -rf data/parquet/* __pycache__ .pytest_cache
	find . -name '*.pyc' -not -path './venv/*' -delete
