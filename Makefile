.PHONY: help venv ingest eval test clean demo-state snapshot

help:
	@echo "Targets:"
	@echo "  venv       - create local python venv and install requirements"
	@echo "  ingest     - convert data/raw/*.csv -> data/parquet/ (see src/etl/ingest.py)"
	@echo "  eval       - honest out-of-fold ROC/PR-AUC report (src/eval/run_auc.py)"
	@echo "  test       - run pytest"
	@echo "  demo-state - regenerate data/snapshots/ feedback-loop artifacts (consent + escalation)"
	@echo "  snapshot   - demo-state + write the bundled UI snapshot (src/ui/assets/worklist.sample.json)"
	@echo "  clean      - remove generated parquet + caches"

# Regenerate the feedback-loop artifacts in data/snapshots/ from scratch. The
# escalation clock is anchored near today (21d/tick x 5 = 84 days) so the 365-day
# consent validity window stays fresh -- a far-future simulated date would mark
# every consent stale and gate everyone (see escalation_job's startup guard).
demo-state:
	rm -f data/snapshots/escalation_state.json data/snapshots/loop_outcomes.json
	./venv/bin/python -m src.routing.consent
	./venv/bin/python -m src.sync.escalation_job --max-ticks 5 --interval 0 --simulate-days-per-tick 21 --batch-size 3500

# Full end-to-end: regenerate state, then capture the /worklist payload into the
# bundled UI snapshot the static site ships with.
snapshot: demo-state
	./venv/bin/python -m src.api.dump_snapshot

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
