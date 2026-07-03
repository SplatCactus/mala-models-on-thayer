# Model Card — BP Cascade RI Persistence-Risk Model

Last updated: 2026-07-02 (pivoted to binary classification; current primary model is `shallow_random_forest`)

## Pivot: binary classification (2026-07-02)

Team decision: go with Chris's classification framing (stratified K-fold CV,
Logistic Regression / shallow Random Forest, PR-AUC/F1) instead of continuing
the survival-analysis approach below. Two things made this the right call,
not just a stylistic preference:

1. **Sample size.** The incident-user cohort fix (washout + baseline BP
   eligibility, `build_features.py`) shrank the cohort to 115–136 patients
   depending on exact eligibility criteria at time of build — too small for
   a single held-out temporal split to give a stable estimate (the original
   262-patient survival run already hit 1 event in its val split).
2. **It sidesteps the survival duration proxy for free.** The old approach
   needed an approximated time-to-event because `pdc.py` has no real
   gap-onset timestamp (see limitation in the historical section below). A
   plain classifier just uses `has_30_day_gap` directly.

`src/models/survival.py` and `src/models/splits.py` are kept as-is
(unused by the current pipeline) — valid infrastructure for if/when the
cohort is large enough to support a temporal split again (e.g. once the
300K-patient SyntheticRI set is in play). `src/models/common.py` now holds
the shared leakage-rule enforcement (`select_feature_columns`) both
`survival.py` and `classifier.py` depend on, so there is exactly one place
that rule can break, not one per model family.

**Known-leakage note**: results below were trained after fixing a real bug
in `src/features/trajectories.py` (the BP lookback window had lost its
lower bound and used `<=` instead of `<`, so same-day-as-index-date BP
readings and unbounded lookback were leaking into "pre-treatment baseline"
features — confirmed by `tests/test_leakage.py` failing 2/10 before the fix,
and by a suspiciously high PR-AUC that dropped once corrected). All numbers
below are post-fix.

**This is on the 1K-derived incident-user cohort (n=115), not yet the
300K-patient set** — Chris is running the 300K build next; treat these as
the last checkpoint before that, not a final result.

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

- **`src/models/classifier.py`** — the current pipeline. Loads the full
  cohort directly (no split file — K-fold re-splits at training time),
  builds Logistic Regression and shallow Random Forest pipelines
  (median-impute + `class_weight="balanced"`), and runs stratified K-fold CV
  reporting PR-AUC/F1 per fold plus mean/std. Fails loudly (`ValueError`) if
  the minority class has fewer patients than `n_splits`, rather than
  surfacing sklearn's less specific error.
- **`src/models/common.py`** — `select_feature_columns`/`save_model`/
  `load_model`, extracted out of `survival.py` so `classifier.py` doesn't
  transitively depend on scikit-survival for one shared function. Single
  place the feature/outcome leakage rule is enforced for every model family.
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

## Results — classification models (current)

Cohort: **115 patients** (incident-user, post-leakage-fix), **51.3% positive**
(`has_30_day_gap`, 59/115) — a near-balanced label, unlike the older
262-patient cohort's 78/22 split. 5-fold stratified CV (`run_stratified_cv`),
fresh model per fold, no shared fitted state across folds.

| model | PR-AUC | F1 |
|---|---|---|
| no-skill baseline (positive rate) | 0.513 | — |
| Logistic Regression (L2, C=0.1, `class_weight="balanced"`) | 0.625 ± 0.079 | 0.618 ± 0.101 |
| Shallow Random Forest (`max_depth=4`, `class_weight="balanced"`) | **0.657 ± 0.071** | **0.687 ± 0.056** |

Both models clear the no-skill baseline with some margin — the first result
in this project that isn't statistically indistinguishable from chance.
Shallow Random Forest is the better of the two on both metrics and is the
current primary model (`models/shallow_random_forest.joblib`).

**Caveats before reading too much into this:**
- n=115 with 5-fold CV means each fold's validation set is ~23 patients —
  the ±0.07-0.10 spread across folds reflects that; treat the point estimate
  as directionally right, not precise to the second decimal.
- This cohort is now "new antihypertensive users with recent BP vitals
  coverage" specifically (per the incident-user/baseline-eligibility fix) —
  narrower than the original treated-hypertensive population. Results may
  not generalize to chronic/legacy patients who were filtered out.
- Feature set is BP trajectory (`sbp_*`/`dbp_*`, now 100% and 78% complete
  respectively, up from ~5%/~1% pre-fix) plus SDOH flags and age —
  demographics are still held aside from the model, same as survival.py.

## Historical: survival-analysis baseline (superseded, kept for reference)

Trained on the earlier 262-patient cohort, before the incident-user cohort
fix and before the team's classification pivot. Not maintained further, but
`survival.py`/`splits.py` remain in the repo since they're valid
infrastructure for a future larger cohort.

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
2. **Survival duration is a proxy, not a true event time** — applies to
   `survival.py` only (historical/superseded); `classifier.py` doesn't need
   this since it uses `has_30_day_gap` directly. `pdc.py` only reports
   *whether* a gap occurred in the 180-day window, not the day it started.
   `build_survival_labels` (in `survival.py`) approximates duration as
   `pdc_180d * 180` for patients with a gap, and the full window for
   censored patients. Documented in-code as an approximation.
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

**Current (classifier.py)**: PR-AUC and F1 via stratified K-fold CV, against
a no-skill (positive-rate) baseline for context — chosen over accuracy given
the class balance is close to 50/50 now but wasn't guaranteed to be, and
PR-AUC/F1 were Chris's explicit ask.

**Historical (survival.py)**: Harrell's concordance index and cumulative/
dynamic time-dependent AUC. Integrated Brier score was intentionally
omitted there — needed an absolute timescale to calibrate against, and the
duration proxy wasn't trustworthy enough for that.

## Edge cases handled

- `splits.py`/`survival.py`: empty cohorts, all-missing-`index_date`
  cohorts, and 1-2-patient cohorts split without crashing. A split where
  every patient has an unlabeled outcome, or has zero observed events
  (all-censored), raises a clear `ValueError` naming the split and patient
  count instead of a raw sklearn/sksurv stack trace.
- `classifier.py`: `run_stratified_cv` raises `ValueError` naming the actual
  class counts if the minority class has fewer patients than `n_splits`,
  instead of surfacing sklearn's less specific error.

## Artifacts

Current: `models/logistic_regression.joblib` and
`models/shallow_random_forest.joblib` (gitignored, produced by
`python src/models/classifier.py`; each refit on the full 115-patient
cohort after CV evaluation — the fold-level fits are not what's saved).

Historical: `models/survival_rsf.joblib` (gitignored, produced by
`python src/models/survival.py`). Frozen split assignment:
`data/snapshots/splits.parquet` (gitignored, produced by
`python src/models/splits.py`).
