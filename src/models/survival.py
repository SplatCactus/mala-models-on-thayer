"""survival.py — tree-based survival model for antihypertensive persistence.

Consumes the artifacts produced upstream:

    feature_panel.parquet   (X) -- src/features/build_features.py
    labels.parquet          (y) -- src/features/pdc.py, via build_features.py
    data/snapshots/splits.parquet (train/val/test) -- src/models/splits.py

and fits a RandomSurvivalForest (scikit-survival) predicting time-to-a-
sustained-medication-gap, right-censored at the end of the fixed follow-up
window.

Leakage rules enforced here (row-level pre/post index_date filtering is
already enforced inside src/features/*.py -- this module enforces the
feature/outcome separation at the model-input boundary):

  1. Outcome columns (``pdc_180d``, ``has_30_day_gap``) must never end up in
     the feature matrix X. :func:`select_feature_columns` raises
     ``AssertionError`` if they do, instead of silently excluding them --
     this is a hard runtime guard, not just a naming convention.
  2. Demographic columns attached in src/etl/cohort.py "for equity strata"
     (RACE, ETHNICITY, GENDER, CITY, STATE, ZIP, LAT, LON,
     HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE, INCOME, is_deceased) are held
     aside from X by construction -- only an explicit allowlist of
     clinical/SDOH columns (the ``sbp_*``/``dbp_*``/``flag_*`` prefixes plus
     ``age_years``) is ever selected as a feature. This matches cohort.py's
     stated intent that those demographics are for subgroup fairness
     auditing, not model inputs, even though build_features.py currently
     merges them straight into feature_panel.parquet alongside the real
     features.

KNOWN LIMITATION -- read before trusting concordance/risk numbers
------------------------------------------------------------------
pdc.py reports only *whether* a >=30-day gap occurred somewhere inside the
fixed ``[index_date, index_date + 180d)`` window, plus the overall coverage
fraction (``pdc_180d``) -- it does not report the day a gap started. A true
survival duration (time from index_date to gap onset) is therefore not
available without changing pdc.py, which is out of scope here.

As a stand-in, :func:`build_survival_labels` derives duration from
``pdc_180d``: censored patients (no gap observed) get the full window
length (``FORWARD_DAYS``, fully observed); patients with an observed gap
get ``pdc_180d * FORWARD_DAYS`` (their total covered days in the window) as
an approximate "time survived before therapy failed". This is only a proxy,
not the true onset day -- coverage can resume after a gap closes, so two
patients with the same covered-day total may have failed at different
points in the window. It is used instead of a constant duration because
scikit-survival's tree splitting (log-rank test) and
``RandomSurvivalForest.predict`` require real variation in event times;
giving every patient an identical duration crashes prediction (confirmed
while building this module -- see git history) rather than merely
degrading gracefully to a classifier. Replace this proxy with a real
gap-onset day once pdc.py is extended to emit one.

Separately, pdc.py has no concept of death: a patient who dies mid-window
and therefore stops refilling is currently labeled event=1 (gap) rather than
censored, because src/etl/cohort.py drops DEATHDATE before the cohort
snapshot reaches this pipeline. Neither limitation is fixed here since doing
so requires changes in src/features/ and src/etl/, out of scope for this
module.

Public API
----------
select_feature_columns(panel_df, ...) -> list[str]
build_survival_labels(labels_df, ...) -> structured ndarray (event, time)
load_training_frame(panel_path, labels_path, splits_path, split_name, ...)
    -> (X, y, patient_ids)
fit_survival_model(X_train, y_train, **model_params) -> fitted sklearn Pipeline
predict_risk(model, X) -> ndarray
predict_survival_function(model, X) -> ndarray of step functions
evaluate_survival_model(model, X, y) -> dict
evaluate_time_dependent_auc(model, y_train, X_eval, y_eval, times) -> dict
save_model(model, path) / load_model(path)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "feature_panel.parquet"
LABELS_PATH = ROOT / "labels.parquet"
SPLITS_PATH = ROOT / "data" / "snapshots" / "splits.parquet"
MODEL_OUTPUT_PATH = ROOT / "models" / "survival_rsf.joblib"

# Must match src/features/pdc.py's forward_days default -- the fixed
# follow-up window every patient is observed over. See module docstring
# "KNOWN LIMITATION" for why this is a constant rather than a per-patient
# value.
FORWARD_DAYS = 180

# Outcome columns from labels.parquet (src/features/pdc.py). These must
# never appear among the selected feature columns.
OUTCOME_COLUMNS = frozenset({"pdc_180d", "has_30_day_gap"})

# Identifier / non-feature columns.
NON_FEATURE_ID_COLUMNS = frozenset({"patient_id", "index_date", "Id", "split"})

# Demographic columns attached in src/etl/cohort.py "for equity strata" --
# held aside from the model by design (see module docstring). Not an
# exhaustive schema validation, just the columns cohort.py is known to emit.
HELD_ASIDE_DEMOGRAPHIC_COLUMNS = frozenset({
    "RACE", "ETHNICITY", "GENDER", "CITY", "STATE", "ZIP", "LAT", "LON",
    "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE", "INCOME", "is_deceased",
})

# Clinical/SDOH feature prefixes emitted by src/features/trajectories.py
# (sbp_*, dbp_*) and src/features/sdoh.py (flag_*).
FEATURE_PREFIXES = ("sbp_", "dbp_", "flag_")

# Non-prefixed columns explicitly allowed as features.
EXPLICIT_FEATURE_COLUMNS = frozenset({"age_years"})

DEFAULT_MODEL_PARAMS = dict(
    n_estimators=200,
    min_samples_leaf=15,
    max_features="sqrt",
    n_jobs=-1,
    random_state=0,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("survival")


def select_feature_columns(
    panel_df: pd.DataFrame,
    *,
    patient_id_col: str = "patient_id",
) -> list[str]:
    """Pick the modeling feature columns out of a merged panel+labels frame.

    Only columns matching :data:`FEATURE_PREFIXES` or listed in
    :data:`EXPLICIT_FEATURE_COLUMNS` are selected. Identifier columns
    (:data:`NON_FEATURE_ID_COLUMNS`), outcome columns
    (:data:`OUTCOME_COLUMNS`), and held-aside demographics
    (:data:`HELD_ASIDE_DEMOGRAPHIC_COLUMNS`) are always excluded.

    Raises
    ------
    AssertionError
        If an outcome column ever ends up selected -- this is the explicit
        code-level enforcement of the feature/outcome leakage rule, not just
        a naming convention.
    """
    selected = [
        col for col in panel_df.columns
        if col != patient_id_col
        and col not in NON_FEATURE_ID_COLUMNS
        and col not in OUTCOME_COLUMNS
        and col not in HELD_ASIDE_DEMOGRAPHIC_COLUMNS
        and (col.startswith(FEATURE_PREFIXES) or col in EXPLICIT_FEATURE_COLUMNS)
    ]

    leaked = OUTCOME_COLUMNS & set(selected)
    if leaked:
        raise AssertionError(
            f"outcome column(s) {sorted(leaked)} selected as model features -- leakage"
        )
    return selected


def build_survival_labels(
    labels_df: pd.DataFrame,
    *,
    event_col: str = "has_30_day_gap",
    coverage_col: str = "pdc_180d",
    forward_days: int = FORWARD_DAYS,
):
    """Build a scikit-survival structured (event, time) array from pdc.py's labels.

    See the module docstring's "KNOWN LIMITATION" section for why ``time``
    is a ``pdc_180d``-derived proxy rather than a true gap-onset day:
    censored patients (``event_col`` == 0) get the full window
    (``forward_days``); patients with an observed gap get
    ``coverage_col * forward_days`` (their covered-day count), clipped to
    at least 1 day since scikit-survival requires strictly positive times.

    Raises
    ------
    ValueError
        If ``event_col`` or ``coverage_col`` contains nulls -- a patient
        with an unlabeled outcome (e.g. missing index_date in pdc.py) must
        be dropped by the caller before reaching this function, not
        silently coerced.
    """
    if labels_df[event_col].isna().any():
        n_null = int(labels_df[event_col].isna().sum())
        raise ValueError(
            f"{n_null} row(s) have a null {event_col!r} -- drop unlabeled "
            "patients before building survival labels"
        )
    if labels_df[coverage_col].isna().any():
        n_null = int(labels_df[coverage_col].isna().sum())
        raise ValueError(
            f"{n_null} row(s) have a null {coverage_col!r} -- drop unlabeled "
            "patients before building survival labels"
        )
    event = labels_df[event_col].astype(bool).to_numpy()
    covered_days = labels_df[coverage_col].clip(lower=0.0, upper=1.0).to_numpy() * forward_days
    time = np.where(event, covered_days, float(forward_days))
    time = np.clip(time, a_min=1.0, a_max=None)
    return Surv.from_arrays(event=event, time=time)


def load_training_frame(
    panel_path: Path,
    labels_path: Path,
    splits_path: Path,
    split_name: str,
    *,
    patient_id_col: str = "patient_id",
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Load X, y, and patient_ids for one split, joined on patient_id.

    Parameters
    ----------
    split_name : str
        One of "train", "val", "test" -- must match a value produced by
        :func:`src.models.splits.make_temporal_splits`.

    Returns
    -------
    X : pandas.DataFrame
        Feature columns only (see :func:`select_feature_columns`).
    y : structured ndarray
        scikit-survival (event, time) array, one row per patient in X.
    patient_ids : pandas.Series
        Patient ids in the same row order as X/y, for joining predictions
        back to the routing/explain stages.
    """
    panel = pd.read_parquet(panel_path)
    labels = pd.read_parquet(labels_path)
    splits = pd.read_parquet(splits_path)

    frame = (
        panel
        .merge(labels, on=patient_id_col, how="inner")
        .merge(splits[[patient_id_col, "split"]], on=patient_id_col, how="inner")
    )
    frame = frame[frame["split"] == split_name].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"no patients assigned to split {split_name!r}")

    n_before = len(frame)
    frame = frame.dropna(subset=["has_30_day_gap", "pdc_180d"]).reset_index(drop=True)
    n_dropped = n_before - len(frame)
    if n_dropped:
        log.warning(
            "  dropping %d/%d patient(s) in split %r with an unlabeled outcome",
            n_dropped, n_before, split_name,
        )
    if frame.empty:
        raise ValueError(
            f"split {split_name!r} had {n_before} patient(s), but all of them have an "
            "unlabeled outcome (missing pdc_180d/has_30_day_gap) -- nothing left to train/evaluate on"
        )

    feature_cols = select_feature_columns(frame, patient_id_col=patient_id_col)
    X = frame[feature_cols].copy()
    y = build_survival_labels(frame)
    patient_ids = frame[patient_id_col].reset_index(drop=True)
    return X, y, patient_ids


def fit_survival_model(X_train: pd.DataFrame, y_train: np.ndarray, **model_params) -> Pipeline:
    """Fit a median-imputed RandomSurvivalForest.

    Imputation is required because BP-trajectory features (sbp_*/dbp_*) are
    NaN for patients with no pre-index readings, and RandomSurvivalForest
    does not accept NaN natively. ``add_indicator=True`` keeps missingness
    itself available as a signal (a patient with no pre-index BP readings is
    informative, not just an inconvenience).

    Raises
    ------
    ValueError
        If ``y_train`` has zero observed events (every patient censored --
        e.g. an unlucky/small temporal split where nobody in "train" has a
        30-day gap). RandomSurvivalForest cannot fit without at least one
        event; this raises with the actual patient count instead of
        surfacing sksurv's generic "all samples are censored" deep in a
        pipeline traceback.
    """
    n_events = int(y_train["event"].sum())
    if n_events == 0:
        raise ValueError(
            f"cannot fit: 0/{len(y_train)} training patients have an observed event "
            "(has_30_day_gap=1) -- all are censored. Check split sizes/ratios in splits.py."
        )
    params = {**DEFAULT_MODEL_PARAMS, **model_params}
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("rsf", RandomSurvivalForest(**params)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_risk(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Per-patient risk score (higher = higher risk of a sustained gap)."""
    return model.predict(X)


def predict_survival_function(model: Pipeline, X: pd.DataFrame):
    """Per-patient survival function S(t) step functions over the follow-up window."""
    X_imputed = model[:-1].transform(X)
    return model[-1].predict_survival_function(X_imputed)


def evaluate_survival_model(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    """Harrell's concordance index on held-out data.

    Integrated Brier score is intentionally not computed here -- it needs a
    validation-time grid derived from training-set event times, and the
    duration proxy used in :func:`build_survival_labels` (see module
    docstring "KNOWN LIMITATION") isn't trustworthy enough as an absolute
    timescale to calibrate against yet. Concordance (a ranking metric) is
    more robust to that proxy than a calibration metric would be.

    Raises
    ------
    ValueError
        If ``y`` has zero observed events -- concordance is undefined with
        no events to rank against (a small val/test split can land here by
        chance). Raised with the patient count instead of sksurv's generic
        "all samples are censored".
    """
    n_events = int(y["event"].sum())
    if n_events == 0:
        raise ValueError(
            f"cannot evaluate: 0/{len(y)} patients have an observed event -- "
            "concordance index is undefined with no events. Check split sizes/ratios in splits.py."
        )
    risk_scores = predict_risk(model, X)
    c_index, *_ = concordance_index_censored(y["event"], y["time"], risk_scores)
    return {"concordance_index": float(c_index)}


def evaluate_time_dependent_auc(
    model: Pipeline,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    times: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Cumulative/dynamic time-dependent AUC at the given follow-up times.

    ``y_train`` (not the split being evaluated) is required because sksurv
    estimates the censoring distribution (IPCW weighting) from the training
    set's follow-up, not the evaluation set's -- this is separate from
    :func:`evaluate_survival_model`'s concordance index, which needs no such
    reference set.

    ``times`` must fall strictly inside ``(0, max(y_eval["time"]))`` --
    sksurv raises ``ValueError`` otherwise, which is left to surface as-is
    since there's no safe default window given how narrow the duration
    proxy's range can be (see module docstring "KNOWN LIMITATION").

    Returns
    -------
    dict
        ``{"times": times, "auc": per-time AUC array, "mean_auc": float}``.
    """
    risk_scores = predict_risk(model, X_eval)
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_eval, risk_scores, times)
    return {"times": times, "auc": auc, "mean_auc": float(mean_auc)}


def save_model(model: Pipeline, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path | str) -> Pipeline:
    return joblib.load(Path(path))


def main() -> int:
    log.info("Loading training frame (split=train)...")
    X_train, y_train, _ = load_training_frame(PANEL_PATH, LABELS_PATH, SPLITS_PATH, "train")
    log.info("  %d patients x %d features, %d/%d with an observed event",
              *X_train.shape, int(y_train["event"].sum()), len(y_train))

    log.info("Fitting RandomSurvivalForest...")
    model = fit_survival_model(X_train, y_train)

    log.info("Loading validation frame (split=val)...")
    X_val, y_val, _ = load_training_frame(PANEL_PATH, LABELS_PATH, SPLITS_PATH, "val")
    val_metrics = evaluate_survival_model(model, X_val, y_val)
    log.info("  val concordance index: %.4f", val_metrics["concordance_index"])

    log.info("Loading test frame (split=test)...")
    X_test, y_test, _ = load_training_frame(PANEL_PATH, LABELS_PATH, SPLITS_PATH, "test")
    test_metrics = evaluate_survival_model(model, X_test, y_test)
    log.info("  test concordance index: %.4f", test_metrics["concordance_index"])

    # Time-dependent AUC needs times strictly inside the test set's observed
    # follow-up range -- use evenly spaced quantiles of the observed test
    # times rather than a hardcoded grid, since the duration proxy's range
    # depends on the actual pdc_180d distribution in this split.
    max_test_time = float(y_test["time"].max())
    times = np.quantile(y_test["time"][y_test["time"] < max_test_time], [0.25, 0.5, 0.75])
    times = np.unique(times)
    if len(times):
        auc_metrics = evaluate_time_dependent_auc(model, y_train, X_test, y_test, times)
        for t, auc in zip(auc_metrics["times"], auc_metrics["auc"]):
            log.info("  test time-dependent AUC @ day %.0f: %.4f", t, auc)
        log.info("  test mean time-dependent AUC: %.4f", auc_metrics["mean_auc"])
    else:
        log.warning("  no valid time points for time-dependent AUC (test set follow-up too narrow)")

    save_model(model, MODEL_OUTPUT_PATH)
    log.info("Saved %s", MODEL_OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
