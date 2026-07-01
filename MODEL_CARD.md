# Model Card — BP Cascade RI Persistence-Risk Model

Last updated: 2026-07-01 (baseline results added after first training run on the 1K cohort)

## What this model does

Predicts each treated-hypertensive patient's risk of a sustained
(>=30-day) antihypertensive medication gap in the 180 days after their
first fill (`index_date`), so the highest-risk patients can be routed to
the appropriate CHW/pharmacist/social-worker intervention (see
`code_dictionary.yaml`'s `routing_action` fields).

## Data

- Cohort: 262 treated-hypertensive patients (`src/etl/cohort.py`) from the
  Synthea 1k synthetic EHR/claims bundle (see `SCHEMA.md`).
- Features (`feature_panel.parquet`, via `src/features/build_features.py`):
  pre-index BP trajectory (`sbp_*`/`dbp_*`, 365-day lookback) and SDOH
  barrier flags (`flag_*`), both strictly before `index_date`.
- Labels (`labels.parquet`, via `src/features/pdc.py`): `pdc_180d`
  (proportion of days covered) and `has_30_day_gap` (binary), both over the
  forward window `[index_date, index_date + 180d)`.

## What was built this session

- **`src/models/splits.py`** — temporal train/val/test split (70/15/15
  default) ordered by `index_date`, not random. A runtime assertion
  (`_assert_no_temporal_leakage`) fails loudly if any split boundary is out
  of chronological order.
- **`src/models/survival.py`** — `RandomSurvivalForest` (scikit-survival) in
  a `SimpleImputer(add_indicator=True) -> RSF` pipeline. Feature selection
  (`select_feature_columns`) uses an explicit allowlist (`sbp_*`, `dbp_*`,
  `flag_*`, `age_years`) and raises `AssertionError` if an outcome column
  ever leaks into it. Demographic columns from `cohort.py`
  (RACE/ETHNICITY/GENDER/ZIP/etc.) are excluded from the model by design —
  held aside for subgroup fairness auditing, not fed to the tree.
- **`tests/test_leakage.py`** — 10 tests proving the index-date leakage rule
  holds: boundary cases (a reading/condition exactly *on* index_date, and
  after) plus randomized property tests across 50 synthetic patients for
  both `trajectories.py` and `sdoh.py`, plus the flip-side check on
  `pdc.py` (a pre-index fill must not count toward the forward coverage
  window). **All 10 passing.**

## Results — baseline trained on the 1K cohort

Temporal split (by `index_date`, via `splits.py`): **183 train / 39 val / 40
test**. Event rate (`has_30_day_gap`) varies sharply by split just from
sample size: **50/183 (27.3%) train, 1/39 (2.6%) val, 6/40 (15.0%) test**.

Held-out **test** set:

- **Concordance index: 0.391**
- **Time-dependent AUC: 0.401 @ day 38, 0.457 @ day 60 (mean 0.435)**

**Read this as "not distinguishable from chance," not as "the model is
anti-predictive."** Both are below 0.5, but the test set has only 6 observed
events (val has 1) — at that sample size a C-index this far from 0.5 is well
within pure sampling noise. This result reflects two compounding constraints,
neither of which this baseline could have modeled around:

1. **Sample size.** 262 patients total, temporally split, leaves single-digit
   event counts in val/test. No amount of model tuning fixes that.
2. **BP-feature sparsity, independent of the index_date fix.** Fixing
   `index_date` to filter to antihypertensive fills specifically (see
   limitation #1 below) corrected clinically-impossible dates but did not
   meaningfully improve BP-trajectory coverage: `sbp_mean`/`dbp_mean` are
   non-null for only 12/262 patients (4.6%), and `sbp_trajectory_slope`/
   `dbp_trajectory_slope` for only 3/262 (1.1%) — both **before and after**
   the fix (13/262 and 0/262, respectively, on the old buggy data). The
   model is effectively learning off SDOH flags and age alone.

**This is not a verdict on the tree-based survival modeling approach** —
it's a verdict on what 262 patients with ~95%-missing vitals can support.
The next useful lever is more data (the 300K-patient SyntheticRI set
`code_dictionary.yaml` already references), not further tuning of this
baseline.

## Known limitations (read before trusting model output)

1. **`index_date` derivation is fixed but still a placeholder.**
   `build_features.py` now correctly filters to each patient's first
   *antihypertensive* fill (not first fill of any medication — that bug is
   fixed), but it's still a TEMP stand-in for a real clinical index date
   (e.g. an HTN diagnosis date). Splits/model treat it as an injected
   parameter, so swapping in a real index date requires no changes here.
2. **Survival duration is a proxy, not a true event time.** `pdc.py` only
   reports *whether* a gap occurred in the 180-day window, not the day it
   started. `build_survival_labels` (in `survival.py`) approximates duration
   as `pdc_180d * 180` for patients with a gap, and the full window for
   censored patients. This is documented in-code as an approximation;
   replace it once `pdc.py` emits a real gap-onset day.
3. **Death is not treated as censoring.** `cohort.py` drops `DEATHDATE`
   before the cohort snapshot is written, so a patient who dies mid-window
   (and therefore stops refilling) is currently labeled as a gap rather
   than censored.
4. **Demographics still flow into `feature_panel.parquet`.** `survival.py`
   excludes them via an allowlist, but `build_features.py` itself doesn't —
   any other consumer of `feature_panel.parquet` needs the same allowlist
   discipline until that's fixed at the source.
5. **No pinned dependencies.** No `requirements.txt` exists in the repo;
   `scikit-learn`, `scikit-survival`, and `joblib` aren't installed in any
   project venv yet (verified missing on this machine — all testing this
   session used a throwaway scratchpad venv).

## Evaluation

Harrell's concordance index (`evaluate_survival_model`) and cumulative/
dynamic time-dependent AUC (`evaluate_time_dependent_auc`, added after the
first baseline run). Integrated Brier score is intentionally omitted — it
needs an absolute timescale to calibrate against, and the duration proxy
above isn't trustworthy enough for that yet (both concordance and
time-dependent AUC are ranking metrics, more robust to the proxy than a
calibration metric would be).

## Edge cases handled (see `survival.py`/`splits.py`)

- Empty cohorts, all-missing-`index_date` cohorts, and 1-2-patient cohorts
  split without crashing.
- A split where every patient has an unlabeled outcome, or where a split
  has zero observed events (all-censored), now raises a clear `ValueError`
  naming the split and patient count — instead of a raw sklearn/sksurv
  stack trace several layers removed from the actual cause.

## Artifacts

Fitted model: `models/survival_rsf.joblib` (gitignored, produced by
`python src/models/survival.py`). Frozen split assignment:
`data/snapshots/splits.parquet` (gitignored, produced by
`python src/models/splits.py`).
