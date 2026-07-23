# BP Cascade RI

🏆 **Winner — RI-AI4H Data Challenge + Datathon 2026, Cardiovascular Disease
& Hypertension track** (Brown University Center for Biomedical Informatics,
July 22–23, 2026).

**Forecasting hypertension medication persistence breaks — and routing the
right helper before they happen.**

Built for the [RI-AI4H Data Challenge 2026](https://bcbi.brown.edu/news-events/ri-ai4h-2026)
(Cardiovascular Disease & Hypertension track), where it won. BP Cascade RI
predicts which treated-hypertensive patients are at risk of a sustained
(≥30-day) antihypertensive medication gap, uses SHAP attribution to surface
each patient's dominant modifiable barrier, and routes them to the right
intervention — pharmacist, social worker, or community health worker call —
under realistic staffing caps.

The output is a capacity-capped, four-language CHW worklist: a name, a
predicted risk, a barrier, and a routed action with a provider-addressed
dispatch message. Provider-facing text ships in English, Spanish, Portuguese,
and Haitian Creole, chosen against Rhode Island household-language data
(`LANGS = ("en", "es", "pt", "ht")` in `src/routing/dispatch_messages.py`).

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
- **Scoped, fail-closed consent.** Rhode Island's CHCCIA
  (R.I. Gen. Laws § 5-37.3-4) is stricter than HIPAA: disclosure to a
  non-covered entity requires the patient's own signed authorization, which an
  Accountable Entity cannot provide on their behalf. The system models
  `internal_care_coordination` and `external_disclosure` as **separate scopes**,
  gates each action against the correct one, and fails closed on missing,
  unknown, or expired authorization — substituting a permitted internal action
  rather than dropping the patient. In the current bundled snapshot
  (`src/ui/assets/worklist.sample.json`) **249** worklist rows carry a gated
  action with an internal fallback recorded (see `src/routing/consent.py`).
- **Leakage discipline:** demographics (race / ethnicity / gender / income / ZIP /
  geo / healthcare-cost) are held out of the model by an allowlist in
  `src/models/common.py`; `tests/test_leakage.py` + `tests/test_pre_index_leakage.py`
  (**31 leakage tests**) enforce the strictly-pre-index feature rule; the full
  suite is **98 tests, all passing** (leakage + escalation, consent, connectors,
  dispatch, API contract, retraining, and an end-to-end integration).
- Routing: 942-patient worklist capped by role capacity
  (CHW call / social worker / pharmacist), with a four-language rationale per patient
- **Synthetic-data ceiling:** Synthea models adherence as a stable per-patient
  trait, so pre-index cross-drug refill behavior predicts post-index antihypertensive
  adherence more cleanly than it would on real EHR data — treat 0.857 as a
  synthetic-data ceiling, not a real-world estimate.

## Recognition

Won the **RI-AI4H Data Challenge + Datathon 2026**, hosted by Brown
University's Center for Biomedical Informatics, **July 22–23, 2026**, in the
**Cardiovascular Disease & Hypertension** track. Judging spanned three
presentation rounds and scored six criteria: clinical importance; community
impact, with emphasis on Hispanic and Latino communities; marketing and
adoption potential; technical feasibility; clarity and coherence of the
proposed solution; and responsiveness to judge feedback. The result rested on
more than a model score — which is where this project is strongest.

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

## The escalation ladder (Rounds 0/1/2)

Routing is the entry point to a timed, latency-adjusted, consent-gated
escalation ladder — server-side and **provider-only** (the system never
messages a patient). See `FEEDBACK_LOOP.md` and `API_CONTRACT.md` for the
full state machine and served shapes.

- **Round 0 — CHW → pharmacy (all patients).** The dispatch goes to the AE's
  CHW with an instruction to contact the pharmacy. We **never** contact a
  pharmacy directly: retail pharmacies will not act on an uncontracted third
  party's alert, and third-party refill steering raises anti-kickback concerns.
- **Round 1 — SDOH-specific (only if no refill).** Routes on the dominant SHAP
  barrier → social worker, pharmacist, bilingual CHW, or a transit voucher (the
  only *external* disclosure, consent-gated).
- **Round 2 — escalate to the AE prescriber (only if Round 1 fails).** The
  dispatch body carries the **complete prior-round history** (every round,
  recipient, date, outcome).

Closure happens **only** on an objective dispense event before the predicted
break date — never on a worker marking a task complete. Timers are
latency-adjusted (`effective_escalate_at = escalate_at + source_max_latency_days`)
so no patient is escalated inside the confirming source's latency window.

**`n_round2` is 0 on synthetic data, and this is structural, not a bug.** The
`local_file` connector derives each patient's outcome from the fixed
`has_30_day_gap` label in `labels.parquet`, so every patient resolves
permanently — `CLOSED` (adherent) or `EXHAUSTED` (confirmed break) — at first
observation, leaving no open "no refill *yet*" interval for the ladder to climb.
Round 2 is fully implemented and unit-tested (`tests/test_escalation*.py`,
`tests/test_dispatch_messages.py`); it needs a longitudinal feed (real AE claims)
with genuinely unresolved patients to fire. We do not fabricate Round 2 patients
to make the funnel look fuller.

## Commercialization

A brief thesis, not signed customers — no pilot exists. See `BUSINESS_CASE.md`
for the cost-avoidance model, the CHCCIA constraint, and pilot prerequisites.

- **Go-to-market thesis:** Rhode Island Medicaid Accountable Entities (AEs) and
  the MCOs that contract them.
- **Why they pay:** Controlling High Blood Pressure is a mandatory quality
  measure. A patient fails it either through uncontrolled pressure or by having
  **no reading on file for the year** — the fall-out-of-care population this
  system targets.
- **Deliberately single-state:** one Medicaid transit broker, one statewide
  referral network, one set of CHW billing rules, one specific language mix. The
  design does not pretend to generalize across states.

**Primary research on the refill signal.** Data-access constraints were
confirmed directly with a Surescripts e-prescribing representative: Surescripts
is e-prescribing *infrastructure*, not a queryable refill feed; medication
history is prescriber-initiated and encounter-tied; and PBMs pass fill data
across the network on request rather than retaining it. This is why AE claims
export — not Surescripts — is the realistic dispense-confirmation source, and
why the connector layer carries per-source latency (`src/sync/connectors/`).

## Layout

```
data/          # raw CSVs, parquet conversions, frozen snapshots — NEVER committed (gitignored)
  raw/         # source CSVs land here (Globus destination)
  parquet/     # ingest.py output
  snapshots/   # routing_table.json, fairness_report.json, auc_report.json,
               #   consent.json, escalation_state.json, loop_outcomes.json,
               #   pharmacy_sync_state.json, retrain_labels/labels_retrained
src/
  etl/         # ingest + treated-hypertensive cohort extraction
  features/    # feature engineering (trajectories, pdc, sdoh, pre_index) + labels
  models/      # classifier.py (current) + common.py leakage allowlist; survival.py/splits.py
               #   kept for a future temporal split; retrain_labels.py harvests closed loops
  explain/     # SHAP attribution (shap_runner.py — scores the deployed model)
  routing/     # routing rules + capacity capping, plus:
    escalation.py         # Rounds 0/1/2 state machine (pure, deterministic, now-injected)
    consent.py            # two-scope, fail-closed CHCCIA consent gate
    dispatch_messages.py  # provider-addressed, four-language dispatch bodies + read-aloud scripts
  eval/        # run_auc.py (out-of-fold ROC/PR-AUC), fairness.py, calibration.py, score_holdout.py
  sync/        # feedback-loop backend:
    escalation_job.py     # ticks the ladder (demo time-compression affordance)
    loop_closure.py       # State 4 objective-refill detection
    sync_job.py           # fit-once / score machinery reused by the jobs
    connectors/           # dispense-data chain: surescripts stub -> ae_claims -> local_file,
                          #   each with per-source latency + access-mode metadata (factory.py)
  api/         # FastAPI worklist + escalation service (main.py); dump_snapshot.py for the bundled UI
  ui/          # four-language CHW dashboard (plain HTML/CSS/JS) + assets/worklist.sample.json
tests/         # leakage audit + escalation, consent, connectors, dispatch, API contract, integration
SCHEMA.md         # data dictionary + cohort validation notes
MODEL_CARD.md     # model documentation, pivot rationale, known limitations
API_CONTRACT.md   # complete read-only worklist + escalation API contract
BUSINESS_CASE.md  # customer, cost-avoidance model, CHCCIA/consent, Surescripts reality, pilot asks
FEEDBACK_LOOP.md  # the four-state closed loop + Rounds 0/1/2 escalation architecture
ESCALATION_PLAN.md # escalation-ladder design notes
Makefile          # common tasks (ingest / eval / test / demo-state / snapshot)
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
