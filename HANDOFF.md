# Handoff: BP-adherence model — honest eval script + feature enrichment

## Context
Project `mala-models-on-thayer`. Predicts antihypertensive **non-adherence** for a
treated-hypertensive cohort. Pipeline: `etl/cohort.py` → `features/build_features.py`
(writes `feature_panel.parquet` = X, `labels.parquet` = y) → `models/classifier.py`
(sklearn logistic + random forest, saved to `models/*.joblib`).

- **Target:** `y = (pdc_180d >= 0.80)`, so `y=1` = *adherent* (85.5% prevalence).
  Non-adherent = the minority we care about.
- **Leakage guard:** `models/common.py::select_feature_columns` — allowlist only
  (`sbp_*`, `dbp_*`, `flag_*`, `age_years`); raises `AssertionError` if an outcome
  column leaks. **All new features must flow through this guard.** Demographics
  (race/ethnicity/income/ZIP) are held aside — do NOT add them as features.
- **Data:** raw CSVs in `data/parquet_300k/*.parquet` (patients, conditions,
  medications, encounters, observations, payer_transitions, etc.). 16,205 labeled
  patients.
- **venv:** `./venv/Scripts/python.exe` — `scikit-learn`, `scipy`, `joblib` were
  missing and have been installed. numpy/pandas/pyarrow/polars already present.

## AUC facts (SUPERSEDED early-exploration numbers — 2026-07 handoff)
> **Superseded (2026-07-23).** The figures in this section are the *early
> exploration* numbers that motivated the GBDT swap, on an earlier/smaller
> feature panel and before final hyperparameter tuning. The current, authoritative
> out-of-fold results live in `data/snapshots/auc_report.json` (reproduce with
> `python src/eval/run_auc.py`) and are summarized in `MODEL_CARD.md`:
> **HGB ROC-AUC 0.857 / PR-AUC(gap) 0.596, logistic 0.681, shallow RF 0.834** on
> the 33-feature panel. Kept below as history; do not quote these as current.
- `classifier.py` reports **PR-AUC** (`average_precision_score`), positive class =
  adherent → prints **0.888/0.904**. That's ~0.855 baseline prevalence + ~0.04 skill.
  Misleading.
- Saved RF **in-sample** ROC-AUC = **0.929** (overfit memorization; `max_depth=12`,
  not the docstring's "4").
- **Honest out-of-fold ROC-AUC (early panel): logistic 0.584, RF 0.626, HGB ~0.648,
  HGB+engineered feats 0.655.** Hispanic vs non-Hispanic gap small and not significant.
  (Now superseded: the enriched 33-feature panel + tuning reach HGB 0.857 — see the
  banner above.)
- 0.9+ is only reachable via outcome leakage (verified: leaked feature → AUC 1.000).
  Do not pursue it. (The ~0.94 figure in `fairness_report.json` is **in-sample**
  scoring of the fitted cohort, not leakage and not a generalization claim.)

## Task 1 — rework `classifier.py` to a gradient-boosted-tree architecture
For tabular EHR data, gradient-boosted decision trees (GBDT) are the correct default —
they handle mixed-scale numeric + binary-flag features, missingness, and non-linear
interactions natively. This is confirmed empirically: out-of-fold ROC-AUC is
**HGB 0.648 vs logistic 0.584 / RF 0.626**. Make GBDT the primary model.

1. Add `build_hist_gradient_boosting(**overrides)` to `classifier.py` returning a
   `Pipeline([("clf", HistGradientBoostingClassifier(...))])`. Starting params:
   `max_depth=6, learning_rate=0.05, max_iter=500, l2_regularization=1.0,
   class_weight="balanced", random_state=0`. **No SimpleImputer/StandardScaler needed** —
   HGB handles NaNs internally and is scale-invariant, so drop those pipeline steps for
   this model (unlike the logistic pipeline, which keeps them).
2. Register it in `MODEL_BUILDERS` and make it the default deployed artifact. Keep
   logistic regression as a lightweight, interpretable baseline; **retire the random
   forest** (`build_shallow_random_forest`) or keep it only as a labeled baseline — it
   is strictly dominated by HGB and its `max_depth=12` overfits (in-sample AUC 0.929 vs
   out-of-fold 0.626).
3. `sklearn`'s `HistGradientBoostingClassifier` needs no new dependency (already
   installed). If you later want `XGBoost`/`LightGBM`, they are viable now — the
   module docstring's "hold off on XGBoost/neural nets until more data" was written for
   the old **n=136** cohort; the current panel has **16,205** patients, so that
   rationale is obsolete. Note that in the docstring if you touch it.
4. Feature columns still route through `common.py::select_feature_columns` — the model
   swap does not change the leakage contract.
5. Re-run Task 2's eval script after the swap to confirm the honest out-of-fold ROC-AUC.

## Task 2 — create a reproducible eval script
Create `src/eval/run_auc.py` (add a `make eval` target). It must:
1. Load via `classifier.load_classification_frame(PANEL_PATH, LABELS_PATH)`.
2. Run **5-fold StratifiedKFold, shuffle=True, random_state=0**, collect
   **out-of-fold** `predict_proba`.
3. Print, per model: **ROC-AUC**, **PR-AUC for the FAILURE class**
   (`average_precision_score(1-y, 1-proba)`) with its no-skill baseline (`1-y.mean()`),
   and **by-ethnicity ROC-AUC** (pull `ETHNICITY` from `feature_panel.parquet`, join on
   `patient_id` in `pids` order).
4. Write results to `data/snapshots/auc_report.json`.

Reference implementation (out-of-fold core):
```python
cv = StratifiedKFold(5, shuffle=True, random_state=0)
p = np.full(len(y), np.nan)
for tr, va in cv.split(X, y):
    m = build(); m.fit(X.iloc[tr], y[tr]); p[va] = m.predict_proba(X.iloc[va])[:, 1]
roc = roc_auc_score(y, p)                         # honest discrimination
pr_fail = average_precision_score(1 - y, 1 - p)   # skill at ranking non-adherers
```
**Never score on training rows** (that's the fake 0.93). Keep in-sample scoring out.

## Task 3 — add features (the real lever; ~0.65 → hopefully higher)
Add new pre-index, leakage-safe feature modules under `src/features/`, merge them in
`build_features.py`, and register their column prefixes in `common.py`
FEATURE_PREFIXES/allowlist so the guard passes. **Every feature must be computed
strictly before each patient's `index_date`** (see the `_date < index_date` pattern in
`sdoh.py`/`trajectories.py`). Then re-run the Task 2 eval script to measure lift.

Features to build, biggest lever first:
1. **Comorbidity count** — # distinct pre-index condition codes from
   `conditions.parquet` (`START < index_date`).
2. **Regimen complexity** — # distinct antihypertensive classes / # concurrent meds at
   index from `medications.parquet` (reuse RxNorm sets in `cohort.py`/`pdc.py`).
3. **Prior-adherence history** — fills/coverage in the washout window *before* index
   (not the 180d after — that's the label).
4. **Engagement proxy** — # encounters (`encounters.parquet`) and # BP readings in the
   pre-index lookback.
5. **Payer churn** — # payer switches from `payer_transitions.parquet` before index.

After each addition, re-run `run_auc.py` and log the ROC-AUC delta so lift is
attributable per feature.

## Guardrails
- **Do not** add demographics as features, evaluate in-sample, or leak the outcome to
  inflate AUC. Report honest out-of-fold numbers even if they stay ~0.65.
- If you keep `build_shallow_random_forest`, **fix its docstring** (says `max_depth=4`,
  code sets `12`) — pick one and make them agree. Preferably retire it (see Task 1).
- The GBDT swap buys ~+0.02–0.06 ROC-AUC — real but modest. Do not expect it to reach
  0.9; architecture is not the bottleneck. **Feature richness (Task 3) and data realism
  are the real levers.**
- Synthea is synthetic; ~0.65 may be the real ceiling. Feature richness helps; the
  honest ceiling is a data-realism limit, not a modeling bug. Say so if lift plateaus.
