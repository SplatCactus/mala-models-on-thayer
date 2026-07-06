# BP Cascade RI

**Forecasting hypertension medication persistence breaks — and routing the
right helper before they happen.**

Built for the [RI-AI4H Data Challenge 2026](https://bcbi.brown.edu/news-events/ri-ai4h-2026)
(Cardiovascular Disease & Hypertension track). BP Cascade RI predicts which
treated-hypertensive patients are at risk of a sustained (≥30-day)
antihypertensive medication gap, uses SHAP attribution to surface each
patient's dominant modifiable barrier, and routes them to the right
intervention — pharmacist, social worker, or bilingual community health
worker call — under realistic staffing caps.

The output is a capacity-capped, bilingual CHW worklist: a name, a
predicted risk, a barrier, and a routed action with an EN/ES outreach
script.

**Dataset:** [SyntheticRI](https://doi.org/10.26300/g7zj-m980) — a
Synthea-generated synthetic EHR dataset of 300,000 patients reflecting
Rhode Island demographics, via the Brown Digital Repository.

**Team (Mala Models on Thayer):** Andres (ETL, API, dashboard, lead) ·
Umar (modeling) · Chris (feature engineering) · Alec (clinical code
dictionary) · Annie (SHAP, fairness, routing).

> **Synthetic-data caveat (read this first):** every result in this repo is
> a methodology and workflow demonstration on synthetic data. Synthea
> generates conditions, medications, and SDOH findings from module logic,
> so effect sizes and correlations are not findings about real Rhode
> Islanders. Validation on real RI clinical data is the explicit next step.

## Results snapshot (300K-derived cohort)

- Cohort: **16,205** treated-hypertensive incident users; **2,356** observed
  ≥30-day gap events in the 180-day outcome window
- Primary model: shallow random forest (binary classification of 30-day gap
  risk; see `MODEL_CARD.md` for the July 2 pivot from the survival framing
  and why)
- Discrimination: AUC **0.571** overall (modest — expected on synthetic
  data; the deliverable is the validated workflow, not the point estimate)
- **Fairness audit:** selection into the capped worklist passes the 80%
  disparate-impact rule (min/max ratio **0.87**); subgroup AUC is slightly
  *higher* for Hispanic patients (0.586 vs 0.568). See
  `data/snapshots/fairness_report.json`
- **Leakage discipline:** `tests/test_leakage.py` caught a real lookback
  bug in BP trajectory features during development; all reported numbers
  are post-fix (documented in `MODEL_CARD.md`)
- Routing: 942-patient worklist capped by role capacity
  (CHW call / social worker / pharmacist), with EN/ES rationale per patient

## Quickstart — run the demo

```bash
# 1. Environment
python3 -m venv venv
source venv/bin/activate
pip install polars pyarrow fastapi uvicorn pandas pyyaml scikit-learn shap

# 2. Serve the API (reads the committed routing snapshot)
./venv/bin/python -m uvicorn src.api.main:app --reload --port 8000

# 3. Serve the dashboard (separate terminal)
cd src/ui && python3 -m http.server 5500
# then open http://localhost:5500/dashboard.html
```

The dashboard fetches `http://localhost:8000/worklist`, which serves
`data/snapshots/routing_table.json` translated into the UI's flat row
format (see `src/api/main.py`).

## Rebuilding from raw data

```bash
make ingest        # raw CSVs -> data/parquet/ (Polars streaming)
python src/etl/cohort.py            # treated-hypertensive cohort
python src/features/build_features.py   # feature panel + labels
python -m src.run_routing_pipeline      # model -> SHAP -> routing table
make test          # includes the leakage audit (tests/test_leakage.py)
```

Raw data is pulled via Globus from the Brown Digital Repository (see the
challenge's transfer guide). `data/` is gitignored — only the small
handoff artifacts (`feature_panel.parquet`, `labels.parquet`, routing
snapshots) are tracked, via explicit `.gitignore` exceptions.

## Layout

```
data/          # raw CSVs, parquet conversions, frozen snapshots — NEVER committed (gitignored)
  raw/         # source CSVs land here (Globus destination)
  parquet/     # ingest.py output
  snapshots/   # frozen routing_table.json + fairness_report.json
src/
  etl/         # ingest + cohort extraction
  features/    # feature engineering (PDC, BP trajectories, SDOH flags)
  models/      # training / inference (classifier.py is current; survival.py kept for future work)
  explain/     # SHAP attribution
  routing/     # routing rules + capacity capping
  eval/        # calibration + fairness
  api/         # FastAPI worklist service
  ui/          # bilingual CHW dashboard (plain HTML/CSS/JS)
tests/         # leakage audit lives here
SCHEMA.md      # data dictionary + cohort validation notes
MODEL_CARD.md  # model documentation, pivot rationale, known limitations
Makefile       # common tasks
```

## Key design commitments

1. **Routed action over flag.** SHAP decomposition of the risk model *is*
   the routing logic — no separately-trained "barrier classifier" inventing
   labels the synthetic data can't support.
2. **Capacity honesty.** The worklist is capped by role headcount so the
   tool respects staffing reality instead of producing an unusable
   firehose.
3. **Leakage discipline.** Features come strictly from the pre-index
   window; the outcome lives strictly in the forward window;
   `tests/test_leakage.py` enforces it.
4. **Synthetic-data honesty.** Method demonstration, not epidemiology.
