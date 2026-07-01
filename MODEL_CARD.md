# Model Card — BP Cascade RI Persistence-Risk Model

Last updated: 2026-07-01

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

## Known limitations (read before trusting model output)

1. **`index_date` is a placeholder.** `build_features.py` currently derives
   it as each patient's first medication fill (a TEMP hack, not a real
   clinical index date). Splits/model treat it as an injected parameter, so
   swapping in a real index date requires no changes here.
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

Harrell's concordance index only. Integrated Brier score is intentionally
omitted — it needs an absolute timescale to calibrate against, and the
duration proxy above isn't trustworthy enough for that yet (concordance is
a ranking metric, more robust to the proxy).

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
