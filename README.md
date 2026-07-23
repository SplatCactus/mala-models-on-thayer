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

Every figure below is reproducible by running the code as it stands:
`python src/eval/run_auc.py` → `data/snapshots/auc_report.json` (the single
source of truth for discrimination), and `python -m src.run_routing_pipeline`
→ the routing + fairness snapshots.

- Cohort: **16,205** treated-hypertensive incident users; **2,362** observed
  ≥30-day gap events in the 180-day outcome window (14.6% event rate)
- Primary model: **HistGradientBoostingClassifier** (gradient-boosted trees) on a
  33-feature leakage-safe panel. Logistic regression is kept as an interpretable
  baseline. The SHAP explainer that scores the routing pipeline uses this same
  primary model, so `routing_table.json` reflects the strong model (see
  `MODEL_CARD.md` for family, hyperparameters, and pivot history).
- Discrimination (honest, strictly out-of-fold, 5-fold stratified CV): ROC-AUC
  **0.857** for the primary model vs **0.681** for logistic regression; PR-AUC for
  the gap event **0.596** against a 0.146 no-skill baseline. (This replaces the
  earlier **0.571** figure, which came from a weak 10-feature logistic model that
  used to score the pipeline and has since been removed from the scoring path.)
- **Fairness audit:** selection into the capped worklist passes the 80%
  disparate-impact rule (min/max selection-rate ratio **0.9933**); out-of-fold
  subgroup ROC-AUC is comparable across groups (Hispanic **0.846** vs
  non-Hispanic **0.859**). See `data/snapshots/fairness_report.json`. Note: that
  file's *overall* AUC field is **in-sample** (the deployed pipeline scores the
  cohort it fit, and a tree model memorizes training rows), so it is not a
  generalization claim — the honest number is the out-of-fold **0.857** above.
- **Leakage discipline:** demographics (race / ethnicity / gender / income / ZIP /
  geo / healthcare-cost) are held out of the model by an allowlist in
  `src/models/common.py`; `tests/test_leakage.py` + `tests/test_pre_index_leakage.py`
  (**31 leakage tests**) enforce the strictly-pre-index feature rule; the full
  suite is **98 tests, all passing** (leakage + escalation, consent, connectors,
  dispatch, API contract, retraining, and an end-to-end integration).
- Routing: 942-patient worklist capped by role capacity
  (CHW call / social worker / pharmacist), with EN/ES rationale per patient
- **Synthetic-data ceiling:** Synthea models adherence as a stable per-patient
  trait, so pre-index cross-drug refill behavior predicts post-index antihypertensive
  adherence more cleanly than it would on real EHR data — treat 0.857 as a
  synthetic-data ceiling, not a real-world estimate.

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
format (see `src/api/main.py`). With the API stopped it falls back to the
bundled snapshot `src/ui/assets/worklist.sample.json`.

### Regenerate the demo state / bundled snapshot

```bash
make demo-state   # regenerate data/snapshots/ feedback-loop artifacts
                  #   (deletes stale escalation_state/loop_outcomes, writes synthetic
                  #    consent.json, then runs the escalation job)
make snapshot     # demo-state + capture GET /worklist into the bundled UI snapshot
                  #   (src/ui/assets/worklist.sample.json)
```

Both anchor the escalation clock near today (21 days/tick × 5 = 84 days).
**Do not run the escalation job with a far-future simulated date:** consent has a
365-day validity window (`CONSENT_VALIDITY_DAYS`) measured against the *simulated*
clock, so a far-future run marks every consent record stale and gates the whole
cohort. `src/sync/escalation_job.py` emits a loud startup warning if the simulated
timeline would exceed that window.

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
