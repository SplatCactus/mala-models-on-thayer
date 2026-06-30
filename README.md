# bp-cascade-ri

Datathon project. Synthea-style EHR/claims data → features → models → explanations → routing.

## Layout

```
data/          # raw CSVs, parquet conversions, frozen snapshots — NEVER committed (gitignored)
  raw/         # source CSVs land here (Globus destination)
  parquet/     # ingest.py output
  snapshots/   # frozen feature_panel
src/
  etl/         # ingest + cleaning
  features/    # feature engineering
  models/      # training / inference
  explain/     # model explanations
  routing/     # routing logic
  eval/        # evaluation
  api/         # service layer
  ui/          # frontend
tests/
SCHEMA.md      # data dictionary
Makefile       # common tasks
```

## Setup

```bash
make venv        # build venv + install deps
make ingest      # CSV -> parquet (once ingest.py is written)
make test
```

Data is never committed — see `.gitignore`.
